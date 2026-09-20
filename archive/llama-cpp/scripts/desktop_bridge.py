"""Private desktop worker. stdin handshake must complete before any GPU work.

The Tauri parent assigns this worker to a kill-on-close Windows Job before
sending a request. All descendants inherit that Job. EOF before the handshake
means the parent died; no inference subprocess is ever started in that case.
"""
from __future__ import annotations

import hashlib
import csv
import io
import json
import logging
import math
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.request

from gpu_config import (ConfigError, choose_binary, detect_gpus,
                        ensure_cache_type_support, memory_plan)
from setup_model import install_model, load_manifest, model_lock, model_url, sha256


def write_json(path: Path, value):
    handle, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    with os.fdopen(handle, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True)
    # Antivirus/readers may briefly hold a file open on Windows.
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                temp.unlink(missing_ok=True)
                raise
            time.sleep(.025)


def validate_settings(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('启动参数必须为对象。')
    result = {'profile': raw.get('profile', 'auto'), 'context': raw.get('context', 2048),
              'parallel': raw.get('parallel', 0), 'ubatch': raw.get('ubatch', 0),
              'port': raw.get('port', 18080), 'cache': raw.get('cache', 'q8_0'),
              'gpu': raw.get('gpu', ''), 'apiKeyEnabled': raw.get('apiKeyEnabled', True),
              'logMode': raw.get('logMode', 'memory'), 'memoryPercent': raw.get('memoryPercent', 30)}
    if type(result['apiKeyEnabled']) is not bool or result['logMode'] not in ('memory', 'file', 'off'):
        raise ValueError('API 密钥开关或日志模式无效。')
    if result['profile'] not in ('auto', 'fast', 'official') or result['cache'] not in ('f16', 'q8_0', 'q4_0'):
        raise ValueError('模型或 KV 缓存类型无效。')
    for key, low, high in [('context', 256, 32768), ('parallel', 0, 128),
                           ('ubatch', 0, 8192), ('port', 1024, 65535), ('memoryPercent', 10, 100)]:
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'{key} 必须是 {low} 至 {high} 之间的整数。')
    if not isinstance(result['gpu'], str) or len(result['gpu']) > 80:
        raise ValueError('显卡编号无效。')
    return result


def reclaim_memory(gpus: list[dict], service: dict) -> list[dict]:
    """Credit only the owned service on its current GPU. Never add its safety
    reserve as reclaimable memory. WDDM often omits per-process VRAM; in that
    case explicitly label the measured startup device delta as an estimate.
    """
    gpu_uuid = service.get('plan', {}).get('gpu', {}).get('uuid')
    reclaimed = max(0.0, float(service.get('loaded_memory_mib', 0)))
    estimated = True
    if service.get('phase') == 'ready' and gpu_uuid:
        try:
            output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=3,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for row in csv.reader(io.StringIO(output.stdout), skipinitialspace=True):
                if len(row) == 3 and row[0].strip() == str(service.get('pid')) and row[1].strip() == gpu_uuid:
                    value = float(row[2].strip())
                    if math.isfinite(value) and value >= 0:
                        reclaimed, estimated = value, False
                    break
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    else:
        reclaimed = 0
    result = []
    for gpu in gpus:
        credit = min(reclaimed, max(0, gpu['total_memory_mib'] - gpu['free_memory_mib'])) if gpu['uuid'] == gpu_uuid else 0
        result.append({**gpu, 'reclaimable_memory_mib': credit, 'reclaim_is_estimate': estimated and credit > 0,
                       'redeploy_available_mib': gpu['free_memory_mib'] + credit})
    return result


def friendly_error(error) -> dict:
    text = str(error)
    low = text.lower()
    hint = '检查运行诊断，修正配置后重试。'
    code = 'runtime'
    for words, kind, advice in [
        (('out of memory', 'cuda error: out', '显存', 'bad_alloc'), 'memory', '降低并发或上下文，关闭其他占用 GPU 的程序；若仅预算比例过低，可调高「显存预算上限」后重新部署。'),
        (('sha256', 'checksum', '大小与清单'), 'integrity', '模型不完整或与当前版本不匹配。请在模型页重新下载，程序会校验 SHA-256。'),
        (('driver', '驱动', 'nvidia-smi', 'no kernel image'), 'gpu', '检查 NVIDIA 显卡和驱动（580.88 或更新），无需另外安装 CUDA Toolkit。'),
        (('download', 'http error', 'urlopen', 'hugging face', 'timed out', 'ssl'), 'network', '检查网络或系统代理后重试；下载临时文件会保留，支持续传。'),
        (('space', '空间', '112'), 'disk', '磁盘空间不足，请清理模型目录所在磁盘后重试。'),
        (('permission', 'access is denied', '拒绝访问'), 'permission', '检查目录写入权限或安全软件；免安装版请放在可写目录。'),
        (('dll', '3221225781', '3221225785'), 'dependency', '运行库缺失或损坏，请重新解压完整免安装包或重新安装。'),
    ]:
        if any(word in low for word in words):
            code, hint = kind, advice
            break
    return {'message': text[-3000:], 'code': code, 'hint': hint}


class Bridge:
    def __init__(self, request: dict):
        self.root = Path(request['root']).resolve()
        self.data = Path(request['data']).resolve()
        self.output = Path(request['output'])
        self.args = request.get('args') or {}
        self.service_state = request.get('service') or {}
        self.models = Path(request.get('model_dir') or self.data / 'models').resolve()
        self.model_registry = Path(request['model_registry']) if request.get('model_registry') else None
        self.search_roots = [Path(p).resolve() for p in request.get('model_search_roots', [str(self.data)]) if p]
        self.models.mkdir(parents=True, exist_ok=True)
        self.model_warnings = []
        for folder in ('logs', 'cache', 'tasks'):
            (self.data / folder).mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(self.root / 'models/manifest.json')

    def model_path(self, item):
        # Identical weights are reused, while different revisions with the same
        # filename can coexist without overwriting another app version's model.
        return self.models / item['sha256'].lower() / item['filename']

    def legacy_model_dirs(self):
        directories = [self.models, self.data / 'models']
        directories.extend(self.registered_model_dirs())
        for seed in self.search_roots:
            for anchor in [seed, *list(seed.parents)[:3]]:
                # Bounded discovery near the app; never recursively scan drives.
                if anchor == anchor.parent:
                    continue
                directories.extend([anchor / 'models', anchor / 'data/models'])
                for pattern in ('HyMT*/data/models', 'HyMT*/models', 'desktop*/HyMT*/data/models'):
                    directories.extend(sorted(anchor.glob(pattern)))
        return list(dict.fromkeys(path.resolve() for path in directories if path.is_dir()))

    def registered_model_dirs(self):
        if self.model_registry:
            try:
                values = json.loads(self.model_registry.read_text('utf-8'))
                if isinstance(values, list):
                    return [Path(p) for p in values[:100] if isinstance(p, str) and Path(p).is_absolute()]
            except (OSError, ValueError):
                pass
        return []

    def remember_model_dir(self):
        # Only an index of paths is shared. Portable weights remain alongside
        # the executable and work after moving the complete folder to a new PC.
        if not self.model_registry:
            return
        try:
            self.model_registry.parent.mkdir(parents=True, exist_ok=True)
            with model_lock(self.model_registry.with_suffix('.lock')):
                previous = self.registered_model_dirs()
                paths = list(dict.fromkeys([self.models, *previous]))[:100]
                if paths != previous:
                    write_json(self.model_registry, [str(p) for p in paths])
        except (OSError, ValueError):
            # A read-only user profile must not break portable operation.
            pass

    def reuse_models(self):
        directories = None
        for item in self.manifest['files'].values():
            target = self.model_path(item)
            if target.exists() or target.with_suffix('.removed').exists():
                continue
            if directories is None:
                directories = self.legacy_model_dirs()
            sources = [source for directory in directories for source in
                       (directory / item['sha256'].lower() / item['filename'], directory / item['filename'])]
            for source in sources:
                if not source.is_file() or source == target:
                    continue
                try:
                    if source.stat().st_size != item['size_bytes'] or sha256(source) != item['sha256'].lower():
                        self.model_warnings.append(f'旧模型与当前清单不符，未复用：{source}')
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with model_lock(target.with_name(target.name + '.lock')):
                        if target.exists():
                            break
                        # Same-volume hard links preserve the original and avoid
                        # doubling disk usage. Cross-volume imports use a copy.
                        try:
                            os.link(source, target)
                        except OSError:
                            part = target.with_name(target.name + '.reuse')
                            try:
                                shutil.copyfile(source, part)
                                if sha256(part) != item['sha256'].lower():
                                    raise ValueError('旧模型复制后的 SHA256 不匹配。')
                                os.replace(part, target)
                            finally:
                                part.unlink(missing_ok=True)
                    break
                except (OSError, ValueError) as error:
                    self.model_warnings.append(f'无法复用 {source}：{error}')
        self.remember_model_dir()

    def verify_model(self, item, path):
        if not path.is_file() or path.stat().st_size != item['size_bytes']:
            raise ConfigError(f'模型缺失或大小与清单不符：{path}。请在模型管理中下载或导入模型。')
        if sha256(path) != item['sha256'].lower():
            raise ConfigError(f'模型 SHA256 与清单不符：{path}。请重新下载或导入模型。')

    def emit(self, value):
        write_json(self.output, value)

    def inventory(self):
        self.reuse_models()
        error = None
        try:
            devices = reclaim_memory(detect_gpus(), self.service_state)
        except (OSError, ValueError) as exc:
            devices, error = [], friendly_error(exc)
        models = []
        for profile, item in self.manifest['files'].items():
            file = self.model_path(item)
            part = file.with_name(file.name + '.part')
            models.append({'profile': profile, **item,
                           'installed': file.is_file() and file.stat().st_size == item['size_bytes'],
                           'downloaded': min(part.stat().st_size, item['size_bytes']) if part.is_file() else 0})
        return {'gpus': devices, 'gpu_error': error, 'models': models,
                'data_dir': str(self.data), 'model_dir': str(self.models),
                'model_warnings': self.model_warnings, 'runtime_dir': str(self.root)}

    def plan(self, verify=False):
        if verify:
            self.reuse_models()
        settings = validate_settings(self.args)
        devices = reclaim_memory(detect_gpus(), self.service_state)
        if settings['gpu']:
            devices = [gpu for gpu in devices if settings['gpu'] in (str(gpu['index']), gpu['uuid'])]
        devices.sort(key=lambda gpu: -gpu['redeploy_available_mib'])
        errors = []
        for gpu in devices:
            try:
                binary, metadata, _ = choose_binary(self.root, 'server', gpu)
                fast = gpu['compute_capability'] in metadata.get('nvfp4_compute_capabilities', [])
                profile = settings['profile']
                if profile == 'auto':
                    profile = 'fast' if fast and gpu['compute_capability'] == '12.0' else 'official'
                if profile == 'fast' and not fast:
                    raise ConfigError('此显卡/构建不支持 NVFP4，请使用 Q4_K_M。')
                item = self.manifest['files'][profile]
                limit = math.floor(gpu['total_memory_mib'] * settings['memoryPercent'] / 100)
                available = min(limit, gpu['redeploy_available_mib'])
                try:
                    plan = memory_plan(mode='server', free_mib=available,
                                       model_bytes=item['size_bytes'], context=settings['context'],
                                       parallel=settings['parallel'] or None, ubatch=settings['ubatch'] or None,
                                       cache_type_k=settings['cache'], cache_type_v=settings['cache'])
                except ConfigError as exc:
                    raise ConfigError(f"显存预算上限为总显存的 {settings['memoryPercent']}%（{limit:.0f} MiB），当前可用于预算 {available:.0f} MiB。{exc} 可降低并发/上下文，或调高显存预算比例。") from exc
                plan['budget'].update(memory_percent=settings['memoryPercent'], memory_limit_mib=limit,
                                      usable_memory_mib=available)
                if verify:
                    self.verify_model(item, self.model_path(item))
                return {**plan, 'gpu': gpu, 'profile': profile, 'port': settings['port'], 'settings': settings,
                        'cache': settings['cache'], 'binary': str(binary),
                        'model': str(self.model_path(item))}
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        raise ConfigError('\n'.join(errors) or '未找到选中的 NVIDIA 显卡。')

    def model_item(self):
        profile = self.args.get('profile')
        if profile not in self.manifest['files']:
            raise ValueError('请指定 fast 或 official 模型。')
        item = self.manifest['files'][profile]
        target = self.model_path(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        return item, target

    def download(self):
        self.reuse_models()
        item, target = self.model_item()
        target.with_suffix('.removed').unlink(missing_ok=True)
        if target.is_file() and target.stat().st_size == item['size_bytes'] and sha256(target) == item['sha256'].lower():
            return {'phase': 'complete', 'profile': self.args['profile']}
        self.emit({'phase': 'downloading', 'profile': self.args['profile']})
        part = target.with_name(target.name + '.part')
        remaining = max(0, item['size_bytes'] - (part.stat().st_size if part.exists() else 0))
        if shutil.disk_usage(target.parent).free < remaining + 128 * 1024**2:
            raise OSError('磁盘空间不足，模型下载还需要约 %.1f GiB。' % (remaining / 1024**3))
        result = install_model(model_url(self.manifest['repo_id'], self.manifest['revision'], item['filename']),
                               target, item['size_bytes'], item['sha256'], timeout=30)
        return {'phase': 'complete', 'result': result, 'profile': self.args['profile']}

    def import_model(self):
        item, target = self.model_item()
        source = Path(self.args.get('path', '')).resolve()
        if source == target.resolve():
            if sha256(source) != item['sha256']:
                raise ValueError('模型 SHA256 不匹配。')
            self.remember_model_dir()
            return {'phase': 'complete'}
        if not source.is_file() or source.stat().st_size != item['size_bytes']:
            raise ValueError('文件大小与清单不符，请导入本项目对应的 fused.gguf 模型。')
        self.emit({'phase': 'verifying', 'profile': self.args['profile']})
        if shutil.disk_usage(target.parent).free < item['size_bytes'] + 128 * 1024**2:
            raise OSError('磁盘空间不足。')
        with model_lock(target.with_name(target.name + '.lock')):
            part = target.with_name(target.name + '.import')
            try:
                shutil.copyfile(source, part)
                if sha256(part) != item['sha256']:
                    raise ValueError('模型 SHA256 不匹配，请选择模型清单对应的文件。')
                os.replace(part, target)
                target.with_suffix('.removed').unlink(missing_ok=True)
            finally:
                part.unlink(missing_ok=True)
        self.remember_model_dir()
        return {'phase': 'complete', 'profile': self.args['profile']}

    def remove_model(self):
        _, target = self.model_item()
        with model_lock(target.with_name(target.name + '.lock')):
            target.with_suffix('.removed').touch()
            target.unlink(missing_ok=True)
            target.with_name(target.name + '.part').unlink(missing_ok=True)
        return {'ok': True}

    def service(self):
        self.emit({'phase': 'starting', 'message': '检查显卡、校验模型并准备推理服务…'})
        plan = self.plan(verify=True)
        port = plan['port']
        for candidate in range(port, min(port + 20, 65536)):
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', candidate))
                    port = candidate
                    break
                except OSError:
                    continue
        else:
            raise OSError('指定端口及后续 19 个端口均被占用，请更改 API 端口。')
        env = os.environ.copy()
        env['PATH'] = os.pathsep.join([str(self.root / 'runtime/cuda'), str(self.root / 'runtime/msvc'), env.get('PATH', '')])
        env.update(CUDA_VISIBLE_DEVICES=plan['gpu']['uuid'], CUDA_CACHE_PATH=str(self.data / 'cache'),
                   LLAMA_HYMT_FUSED_PROJ='1', LLAMA_HYMT_SPARSE_PENALTIES='1',
                   LLAMA_HYMT_SAMPLING_SNAPSHOT='1', LLAMA_HYMT_BATCH_SNAPSHOT='1',
                   GGML_CUDA_HYMT_DISABLE_ROPE_NORM='0', GGML_CUDA_HYMT_ROPE_NORM_STRICT='1',
                   GGML_CUDA_HYMT_EAGER_GRAPHS='1', GGML_CUDA_GRAPH_OPT='0',
                   GGML_CUDA_HYMT_DISABLE_Q8_KV_FUSION='0', GGML_CUDA_Q8_KV_SUBWARP='0')
        settings = plan['settings']
        key_path = self.data / 'api-key.txt'
        key = ''
        if self.args.get('_rotate_key') or (settings['apiKeyEnabled'] and not key_path.exists()):
            key_path.write_text(secrets.token_urlsafe(32), encoding='utf-8')
        if settings['apiKeyEnabled']:
            key = key_path.read_text('utf-8').strip()
            if len(key) < 32 or not all(character.isascii() and (character.isalnum() or character in '-_') for character in key):
                raise ValueError('API 密钥文件无效，请点击 API Key 下方的「刷新密钥」修复并重新部署。')
        # Remove inherited credentials as well when authentication is disabled.
        env.pop('LLAMA_API_KEY', None)
        env.pop('LLAMA_ARG_API_KEY', None)
        if key:
            env['LLAMA_API_KEY'] = key
        exe = Path(plan['binary']) / 'llama-server.exe'
        ensure_cache_type_support(exe, plan['cache'], plan['cache'], env=env, cwd=self.data)
        command = [str(exe), '-m', plan['model'], '--alias', 'hy-mt2', '--host', '127.0.0.1',
                   '--port', str(port), '-ngl', 'all', '-fa', 'on',
                   '--cache-type-k', plan['cache'], '--cache-type-v', plan['cache'],
                   '-c', str(plan['parallel'] * plan['context']), '-np', str(plan['parallel']),
                   '-b', str(plan['batch']), '-ub', str(plan['ubatch']), '-t', '8', '-tb', '8', '--jinja',
                   '--temp', '0.7', '--top-p', '0.6', '--top-k', '20', '--min-p', '0',
                   '--repeat-penalty', '1.05', '--repeat-last-n', str(max(4096, plan['context'])),
                   '--no-context-shift', '--no-warmup', '--no-webui']
        logger = logging.getLogger('inference')
        logger.setLevel(logging.INFO)
        if settings['logMode'] == 'file':
            handler = RotatingFileHandler(self.data / 'logs/inference.log', maxBytes=2*1024**2, backupCount=2, encoding='utf-8')
            logger.addHandler(handler)
        if settings['logMode'] == 'off':
            command.append('--log-disable')
        process = subprocess.Popen(command, env=env, cwd=self.data, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL if settings['logMode'] == 'off' else subprocess.PIPE,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        def log_reader():
            for line in iter(process.stdout.readline, b''):
                text = line.decode('utf-8', errors='replace').rstrip()[:16000]
                if key:
                    text = text.replace(key, '[redacted]')
                if settings['logMode'] == 'file':
                    logger.info(text)
                # Pipe to the desktop's bounded RAM buffer; never put log text
                # into a status file when file logging is disabled.
                print(text, flush=True)
        if settings['logMode'] != 'off':
            threading.Thread(target=log_reader, daemon=True).start()
        ready = False
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 180
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/models', headers={'Authorization': f'Bearer {key}'})
                    with opener.open(request, timeout=1) as response:
                        if json.load(response).get('data'):
                            ready = True
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            if not ready:
                raise RuntimeError(f'推理启动失败/超时（退出码 {process.poll()}），请查看运行诊断。')
            # Secret stays on disk/backend; only returned by an explicit copy action.
            write_json(self.data / 'service-private.json', {'port': port, 'key': key, 'context': plan['context']})
            loaded = 0
            try:
                after = next(gpu for gpu in detect_gpus() if gpu['uuid'] == plan['gpu']['uuid'])
                loaded = max(0, plan['gpu']['free_memory_mib'] - after['free_memory_mib'])
            except (OSError, ValueError, StopIteration):
                pass
            self.emit({'phase': 'ready', 'port': port, 'pid': process.pid, 'plan': plan, 'loaded_memory_mib': loaded,
                       'message': '端口占用，已自动切换。' if port != plan['port'] else ''})
            failures = 0
            while process.poll() is None:
                time.sleep(3)
                try:
                    with opener.open(urllib.request.Request(f'http://127.0.0.1:{port}/health', headers={'Authorization': f'Bearer {key}'}), timeout=3) as response:
                        if json.load(response).get('status') != 'ok':
                            raise OSError('health check failed')
                    failures = 0
                except (OSError, ValueError):
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError('推理服务连续健康检查失败，已停止以释放显存。请查看日志并重新启动。')
            raise RuntimeError(f'推理服务意外退出（{process.returncode}），请查看运行诊断。')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            (self.data / 'service-private.json').unlink(missing_ok=True)

    def log_tail(self):
        path = self.data / 'logs/inference.log'
        if not path.exists():
            return '尚无推理日志。'
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            return stream.read().decode('utf-8', errors='replace')

    def run(self, action):
        methods = {'inventory': self.inventory, 'estimate': self.plan, 'validate_deploy': lambda: self.plan(verify=True), 'service': self.service,
                   'download': self.download, 'import_model': self.import_model,
                   'remove_model': self.remove_model, 'logs': self.log_tail}
        if action not in methods:
            raise ValueError('未知桌面操作。')
        return methods[action]()


def main():
    line = sys.stdin.readline()
    if not line:
        return 0
    request = json.loads(line)
    try:
        bridge = Bridge(request)
        result = bridge.run(request['action'])
        write_json(Path(request['output']), {'ok': True, 'result': result})
        return 0
    except Exception as error:
        write_json(Path(request['output']), {'ok': False, 'phase': 'error', 'error': friendly_error(error)})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
