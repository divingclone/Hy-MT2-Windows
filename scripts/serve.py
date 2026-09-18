"""Portable local Hy-MT server launcher. Uses bundled binaries and stdlib only."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from gpu_config import CACHE_TYPE_BYTES, ensure_cache_type_support, resolve_config

ROOT = Path(__file__).resolve().parents[1]


def process_creation_filetime(process):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    timestamps = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(int(process._handle), *(ctypes.byref(value) for value in timestamps)):
        raise ctypes.WinError(ctypes.get_last_error())
    created = timestamps[0]
    return str((created.dwHighDateTime << 32) | created.dwLowDateTime)


def check_managed_label(label):
    pid_file = ROOT/'results'/f'{label}.pid'
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding='ascii').strip())
        if pid <= 0:
            raise ValueError('invalid PID')
    except ValueError as error:
        raise RuntimeError(f'Invalid managed PID file: {pid_file}') from error
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # PID does not exist.
            return
        raise RuntimeError(f'Cannot verify existing PID {pid} (Windows error {error}); choose another label')
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259:
            raise RuntimeError(f'Label {label!r} already manages a live process (PID {pid}); stop it or choose another label')
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('auto', 'fast', 'official'), default='auto')
    parser.add_argument('--binary-dir', type=Path, help='Explicit executable directory for source-build validation; default: bin')
    parser.add_argument('--model')
    parser.add_argument('--parallel', type=int)
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--ubatch', type=int)
    parser.add_argument('--cache-type-k', choices=CACHE_TYPE_BYTES, default='f16')
    parser.add_argument('--cache-type-v', choices=CACHE_TYPE_BYTES, default='f16')
    parser.add_argument('--gpu')
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--label', default='server')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--backend-sampling', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', args.label):
        parser.error('label must contain only letters, digits, dots, underscores or hyphens')
    if not 1 <= args.port <= 65535:
        parser.error('port must be in 1..65535')
    check_managed_label(args.label)
    config = resolve_config(ROOT, mode='server', profile=args.profile, parallel=args.parallel,
                            context=args.context, ubatch=args.ubatch, gpu=args.gpu,
                            binary_dir=args.binary_dir, model_override=args.model,
                            cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v)
    batch = args.batch or config['batch']
    if batch < config['ubatch']:
        parser.error('batch must be at least ubatch')
    exe = (Path(config['binary_dir'])/'llama-server.exe').resolve()
    if not exe.is_file():
        raise FileNotFoundError(f'Missing server executable: {exe}')
    for directory in ('results', 'cache/cuda', 'cache/llama', 'cache/huggingface'):
        (ROOT/directory).mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    env = os.environ.copy()
    env['PATH'] = str(ROOT/'runtime/cuda') + os.pathsep + str(ROOT/'runtime/msvc') + os.pathsep + env.get('PATH', '')
    env.update(CUDA_VISIBLE_DEVICES=config['cuda_visible_devices'], CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),
               HF_HOME=str(ROOT/'cache/huggingface'), LLAMA_CACHE=str(ROOT/'cache/llama'),
               LLAMA_HYMT_FUSED_PROJ='1', LLAMA_HYMT_SPARSE_PENALTIES='1',
               LLAMA_HYMT_SAMPLING_SNAPSHOT='1', LLAMA_HYMT_BATCH_SNAPSHOT='1',
               GGML_CUDA_HYMT_DISABLE_ROPE_NORM='0', GGML_CUDA_HYMT_ROPE_NORM_STRICT='1',
               GGML_CUDA_HYMT_EAGER_GRAPHS='1', GGML_CUDA_GRAPH_OPT='0',
               GGML_CUDA_HYMT_DISABLE_Q8_KV_FUSION='0', GGML_CUDA_Q8_KV_SUBWARP='0')
    command = [str(exe), '-m', config['model'], '--alias', 'hy-mt2', '--host', '127.0.0.1',
               '--port', str(args.port), '-ngl', 'all', '-fa', 'on',
               '--cache-type-k', args.cache_type_k, '--cache-type-v', args.cache_type_v,
               '-c', str(config['parallel']*args.context), '-np', str(config['parallel']),
               '-b', str(batch), '-ub', str(config['ubatch']), '-t', '8', '-tb', '8', '--jinja',
               '--temp', '0.7', '--top-p', '0.6', '--top-k', '20', '--min-p', '0',
               '--repeat-penalty', '1.05', '--repeat-last-n', str(max(4096, args.context)),
               '--metrics', '--no-context-shift', '--no-warmup']
    if args.backend_sampling:
        command.append('--backend-sampling')
    ensure_cache_type_support(exe, args.cache_type_k, args.cache_type_v, env=env, cwd=ROOT)
    for warning in config['warnings']:
        print(warning, file=sys.stderr)
    config.update(command=command, batch=batch, port=args.port, mode='server', label=args.label, executable=str(exe.resolve()))
    config_path = ROOT/'results'/f'{args.label}.config.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.background:
        return subprocess.call(command, cwd=ROOT, env=env)
    with (ROOT/'results'/f'{args.label}.stdout.log').open('wb') as out, (ROOT/'results'/f'{args.label}.stderr.log').open('wb') as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    (ROOT/'results'/f'{args.label}.pid').write_text(str(process.pid), encoding='ascii')
    config.update(managed_pid=process.pid, process_creation_filetime=process_creation_filetime(process))
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    deadline = time.monotonic() + 120
    local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Server exited ({process.returncode}); inspect results/{args.label}.stderr.log')
        try:
            with local_http.open(f'http://127.0.0.1:{args.port}/health', timeout=1) as response:
                if json.load(response).get('status') == 'ok':
                    print(f"Ready: http://127.0.0.1:{args.port} ; PID {process.pid}; profile={config['profile']}; parallel={config['parallel']}")
                    return 0
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f'Server is still loading; PID {process.pid}; inspect results/{args.label}.stderr.log')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)
