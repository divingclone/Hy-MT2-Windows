"""Verified four-bit checkpoint bundles for the native Windows vLLM backend."""
from __future__ import annotations
import argparse, hashlib, http.client, json, math, os, re, shutil, sys, tempfile, time, zipfile
from contextlib import contextmanager
from pathlib import Path
import urllib.error, urllib.parse, urllib.request
from gpu_config import ConfigError
ROOT=Path(__file__).resolve().parents[1]
CHUNK=4*1024**2
class SetupError(ValueError):
    pass

def model_url(repo: str | None, revision: str, filename: str) -> str:
    if (not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or
            any(part in (".", "..") for part in repo.split("/"))):
        raise SetupError("No valid Hugging Face repository is configured. Use --repo OWNER/REPOSITORY.")
    if not isinstance(revision, str) or not revision or any(c in revision for c in "\r\n"):
        raise SetupError("Model revision is missing or invalid")
    return "https://huggingface.co/" + repo + "/resolve/" + urllib.parse.quote(revision, safe="") + "/" + urllib.parse.quote(filename, safe="")

class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https":
            raise SetupError("Refusing a model download redirect that is not HTTPS")
        redirected = super().redirect_request(request, fp, code, message, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(request.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected

def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

@contextmanager
def model_lock(path: Path):
    # OS locks are released after a crash, so the small persistent lock file does
    # not block a later retry. Never remove a lock path while another process may
    # already have it open.
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise SetupError("Another setup process is downloading this model; wait for it to finish") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def _download_attempt(url: str, part: Path, expected_size: int, opener, timeout: float, token: str | None) -> None:
    offset = part.stat().st_size if part.is_file() else 0
    if offset >= expected_size:
        part.unlink()
        offset = 0
    headers = {"User-Agent": "HyMT-Windows-model-setup/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=timeout) as response:
        if urllib.parse.urlsplit(response.geturl()).scheme.lower() != "https":
            raise SetupError("Model download must use HTTPS")
        status = response.status
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset or int(match[3]) != expected_size or int(match[2]) != expected_size - 1:
                raise SetupError("Unexpected partial-download response; no model was installed")
        elif status == 200:
            offset = 0  # Server ignored Range: restart this controlled partial file.
        else:
            raise SetupError(f"Unexpected download status {status}")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isdigit() or int(length) != expected_size - offset):
            raise SetupError("Download size disagrees with models/manifest.json; no model was installed")
        written = offset
        last_report = 0.0
        with part.open("ab" if offset else "wb") as stream:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                if written + len(chunk) > expected_size:
                    raise SetupError("Download exceeded the expected size; no model was installed")
                stream.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2 or written == expected_size:
                    print(f"\rDownloading: {written / expected_size:6.1%} ({written / 1024**2:.1f} / {expected_size / 1024**2:.1f} MiB)", end="", flush=True)
                    last_report = now
            stream.flush()
            os.fsync(stream.fileno())
        print()
        if written != expected_size:
            raise OSError(f"Download interrupted at {written} of {expected_size} bytes; partial data retained")

def install_model(url: str, destination: Path, expected_size: int, expected_sha256: str,
                  *, retries: int = 3, timeout: float = 60, opener=None, token: str | None = None) -> str:
    if token:
        token = token.strip()
        if any(ord(character) < 33 or ord(character) > 126 for character in token):
            raise SetupError("HF_TOKEN contains invalid characters; set a plain access token")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise SetupError("The model destination must not be a symbolic link")
    expected_sha256 = expected_sha256.lower()
    part = destination.with_name(destination.name + ".part")
    lock = destination.with_name(destination.name + ".lock")
    if part.is_symlink() or lock.is_symlink():
        raise SetupError("Download staging paths must not be symbolic links")
    opener = opener or urllib.request.build_opener(HTTPSRedirectHandler())
    with model_lock(lock):
        if destination.is_file() and destination.stat().st_size == expected_size:
            print("Checking the existing model SHA256...", flush=True)
            if sha256(destination) == expected_sha256:
                return "already_verified"
        if part.is_file() and part.stat().st_size == expected_size and sha256(part) == expected_sha256:
            os.replace(part, destination)
            return "resumed_verified"
        for attempt in range(retries + 1):
            try:
                _download_attempt(url, part, expected_size, opener, timeout, token)
                print("Checking downloaded model SHA256...", flush=True)
                if sha256(part) != expected_sha256:
                    part.unlink()
                    raise OSError("Model SHA256 mismatch; corrupt partial file removed")
                os.replace(part, destination)
                return "downloaded_verified"
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise SetupError("Hugging Face denied access. For a private/gated repository set HF_TOKEN; never paste a token into a command URL.") from None
                if error.code == 404:
                    raise SetupError("Model file was not found on Hugging Face. Check --repo and the published model revision; the repository may not have been uploaded yet.") from None
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                    raise SetupError(f"Hugging Face download failed (HTTP {error.code}); partial data retained") from None
            except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
                if attempt == retries:
                    raise SetupError(f"Download failed after {attempt + 1} attempt(s): {error}. Re-run setup-model.cmd to resume.") from None
            print(f"Retrying download ({attempt + 1}/{retries})...", flush=True)
            time.sleep(min(2**attempt, 8))
    raise SetupError("Model installation did not complete")

def load_manifest(path):
    manifest=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version')!=2 or manifest.get('backend')!='vllm':
        raise SetupError('需要 vLLM schema 2 模型清单，GGUF 不能用于此后端。')
    if not isinstance(manifest.get('files'),dict) or 'fast' not in manifest['files'] or set(manifest['files'])-{'fast','compat'}:
        raise SetupError('模型清单必须包含 fast，可选 compat checkpoint。')
    if set(manifest.get('checkpoints',{})) != set(manifest['files'])-{'fast'}:
        raise SetupError('模型清单的 checkpoint 与模型包不一致。')
    if 'repositories' in manifest:
        if set(manifest['repositories']) != set(manifest['files']):
            raise SetupError('模型仓库与 checkpoint 清单不一致。')
        for remote in manifest['repositories'].values():
            model_url(remote.get('repo_id'),remote.get('revision'),'config.json')
            if not re.fullmatch(r'[a-f0-9]{40}',remote['revision']):
                raise SetupError('模型下载必须固定到完整提交哈希。')
    for profile,item in manifest['files'].items():
        spec=checkpoint_manifest(manifest,profile)
        if not spec.get('checkpoint_files'):
            raise SetupError('模型清单缺少 checkpoint 文件。')
        for name in [spec.get('checkpoint_dir',''),item.get('filename',''),*spec['checkpoint_files']]:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) or name in ('.','..'):
                raise SetupError('模型清单包含不安全的路径。')
        for record in [item,*spec['checkpoint_files'].values()]:
            if type(record.get('size_bytes')) is not int or record['size_bytes']<=0 or not re.fullmatch(r'[a-f0-9]{64}',record.get('sha256','')):
                raise SetupError('模型清单大小或 SHA256 无效。')
        if spec.get('checkpoint_size_bytes') != sum(x['size_bytes'] for x in spec['checkpoint_files'].values()):
            raise SetupError('checkpoint 总大小不符。')
    return manifest


def checkpoint_manifest(manifest,profile):
    variant='compat' if profile=='compat' else 'fast'
    if variant not in manifest['files']:
        raise SetupError(f'运行包清单缺少 {variant} 模型，请更新程序包后再下载模型。')
    if variant=='fast': return manifest
    spec=manifest.get('checkpoints',{}).get(variant)
    if not isinstance(spec,dict): raise SetupError('模型清单缺少 compat checkpoint。')
    return {**manifest, **{k:spec.get(k) for k in ('checkpoint_dir','checkpoint_size_bytes','checkpoint_files')}}

def checkpoint_path(root,manifest):
    return Path(root)/'models'/manifest['checkpoint_dir']

def verify_checkpoint(path,manifest):
    path=Path(path)
    for name,item in manifest['checkpoint_files'].items():
        file=path/name
        if not file.is_file() or file.stat().st_size!=item['size_bytes'] or sha256(file)!=item['sha256']:
            raise SetupError(f'checkpoint 文件缺失或 SHA256 不符：{file}。请运行 setup-model.cmd 或导入对应模型包。')
    return path


def checkpoint_complete(path,manifest):
    """Cheap inventory check; loading still verifies every SHA-256."""
    path=Path(path)
    return bool(manifest['checkpoint_files']) and all(
        (path/name).is_file() and (path/name).stat().st_size==item['size_bytes']
        for name,item in manifest['checkpoint_files'].items())


def checkpoint_downloaded(path,manifest):
    staging=Path(path).with_name(Path(path).name+'.download')
    total=0
    for name,item in manifest['checkpoint_files'].items():
        file=staging/name
        part=file.with_name(name+'.part')
        if file.is_file(): total+=min(file.stat().st_size,item['size_bytes'])
        elif part.is_file(): total+=min(part.stat().st_size,item['size_bytes'])
    return total


def download_checkpoint(target,manifest,profile,*,repo=None):
    """Resume individual raw files, then publish one fully verified directory."""
    target=Path(target)
    remote=manifest.get('repositories',{}).get(profile)
    if not remote: raise SetupError('清单未配置原始模型仓库，请更新程序或导入离线 ZIP。')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.with_name(target.name+'.download')
    if target.is_symlink() or staging.is_symlink(): raise SetupError('模型目录不能是符号链接。')
    with model_lock(target.with_name(target.name+'.lock')):
        if target.exists():
            verify_checkpoint(target,manifest)
            return 'already_verified'
        staging.mkdir(exist_ok=True)
        remaining=manifest['checkpoint_size_bytes']-checkpoint_downloaded(target,manifest)
        if shutil.disk_usage(target.parent).free < remaining+128*1024**2:
            raise SetupError('磁盘空间不足，无法下载模型。')
        for name,item in manifest['checkpoint_files'].items():
            print('Downloading checkpoint file: '+name,flush=True)
            file=staging/name
            install_model(model_url(repo or remote['repo_id'],remote['revision'],name),
                          file,item['size_bytes'],item['sha256'])
            file.with_name(name+'.lock').unlink(missing_ok=True)
        verify_checkpoint(staging,manifest)
        os.replace(staging,target)
    return 'downloaded_verified'

def extract_checkpoint(bundle,target,manifest):
    target=Path(target)
    if target.exists(): return verify_checkpoint(target,manifest)
    target.parent.mkdir(parents=True,exist_ok=True)
    with model_lock(target.with_name(target.name+'.lock')):
        if target.exists(): return verify_checkpoint(target,manifest)
        with tempfile.TemporaryDirectory(prefix='nvfp4-',dir=target.parent) as temporary:
            staging=Path(temporary)/'checkpoint'
            staging.mkdir()
            with zipfile.ZipFile(bundle) as archive:
                names=archive.namelist()
                if len(names)!=len(set(names)) or set(names)!=set(manifest['checkpoint_files']):
                    raise SetupError('模型包文件集合不符；拒绝解压。')
                for name,item in manifest['checkpoint_files'].items():
                    if archive.getinfo(name).file_size!=item['size_bytes']:
                        raise SetupError('模型包解压大小不符。')
                    with archive.open(name) as source,(staging/name).open('wb') as output:
                        shutil.copyfileobj(source,output)
            verify_checkpoint(staging,manifest)
            os.replace(staging,target)
    return target

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='Import the verified ZIP bundle for the selected profile')
    parser.add_argument('--repo',help='Optional raw checkpoint repository with the same pinned revision and files')
    parser.add_argument('--profile',choices=('auto','fast','quality','compat'),default='auto')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--pause-on-exit',action='store_true')
    args=parser.parse_args(argv)
    root=args.root.resolve()
    manifest=load_manifest(root/'models/manifest.json')
    from gpu_config import detect_gpus, select_profile
    profile=args.profile
    if profile=='auto':
        devices=sorted(detect_gpus(),key=lambda x:-x['free_memory_mib'])
        supported=[d for d in devices if d['compute_capability'] in ('8.0','8.6','8.9','12.0')]
        if not supported: raise SetupError('当前运行包需要 RTX 30/40/50 或对应受支持架构。')
        profile=select_profile(supported[0],profile)
    variant='compat' if profile=='compat' else 'fast'
    manifest=checkpoint_manifest(manifest,variant)
    target=checkpoint_path(root,manifest)
    if target.exists():
        verify_checkpoint(target,manifest)
        print(f'Ready: {target}')
        return 0
    item=manifest['files'][variant]
    bundle=args.source or root/'models'/item['filename']
    if not bundle.is_file():
        if args.source: raise SetupError('指定的离线 ZIP 不存在。')
        download_checkpoint(target,manifest,variant,repo=args.repo)
        print(f'Ready: {target}')
        return 0
    if bundle.stat().st_size!=item['size_bytes'] or sha256(bundle)!=item['sha256']:
        raise SetupError('模型包 SHA256 或大小不符。')
    extract_checkpoint(bundle,target,manifest)
    print(f'Ready: {target}')
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError) as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(1)
