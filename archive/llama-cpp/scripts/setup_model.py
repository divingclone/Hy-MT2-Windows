"""Download a verified Hy-MT model using the bundled Python; no pip is needed."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from gpu_config import ConfigError, choose_binary, detect_gpus

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024 * 1024


class SetupError(ValueError):
    pass


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise SetupError(f"Cannot read model manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version", manifest.get("schema")) != 1:
        raise SetupError("models/manifest.json must use schema version 1")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SetupError("Model manifest is missing the files map")
    for profile in ("fast", "official"):
        item = files.get(profile)
        if not isinstance(item, dict):
            raise SetupError(f"Model manifest is missing profile {profile}")
        name, size, digest = item.get("filename"), item.get("size_bytes"), item.get("sha256")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.gguf", name) or
                type(size) is not int or size <= 0 or
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
            raise SetupError(f"Invalid filename, size_bytes or SHA256 for profile {profile}")
    if files["fast"]["filename"].casefold() == files["official"]["filename"].casefold():
        raise SetupError("Profiles must use distinct model filenames")
    return manifest


def model_url(repo: str | None, revision: str, filename: str) -> str:
    if (not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or
            any(part in (".", "..") for part in repo.split("/"))):
        raise SetupError("No valid Hugging Face repository is configured. Use --repo OWNER/REPOSITORY.")
    if not isinstance(revision, str) or not revision or any(c in revision for c in "\r\n"):
        raise SetupError("Model revision is missing or invalid")
    return "https://huggingface.co/" + repo + "/resolve/" + urllib.parse.quote(revision, safe="") + "/" + urllib.parse.quote(filename, safe="")


def select_profile(root: Path, profile: str, gpu: str | None = None) -> tuple[str, str]:
    # An explicit profile can stage a model for another computer without probing
    # local hardware. The inference launcher still validates its actual GPU.
    if profile != "auto" and gpu is None:
        return profile, "explicit profile; hardware is checked when inference starts"
    devices = detect_gpus()
    if gpu is not None:
        selector = str(gpu).strip()
        devices = [item for item in devices if selector in (str(item["index"]), item["uuid"])]
        if not devices:
            raise SetupError(f"GPU {selector!r} was not found; check nvidia-smi -L")
    devices.sort(key=lambda item: (-item["free_memory_mib"], item["index"]))
    failures = []
    for device in devices:
        try:
            _, metadata, _ = choose_binary(root, "batch", device)
            fast_allowed = device["compute_capability"] in metadata.get("nvfp4_compute_capabilities", [])
            if profile == "fast" and not fast_allowed:
                raise SetupError("This GPU/build does not declare the NVFP4 fast profile; use --profile official")
            chosen = ("fast" if fast_allowed and device["compute_capability"] == "12.0" else "official") if profile == "auto" else profile
            return chosen, f"GPU {device['index']}: {device['name']} (CC {device['compute_capability']})"
        except (ConfigError, SetupError) as error:
            failures.append(str(error))
    raise SetupError("No supported GPU/build for automatic setup. Extract the complete runtime package, "
                     "or use --profile official/fast to download explicitly. " + "; ".join(failures))


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Auto: RTX 50 uses NVFP4; other supported GPUs use official Q4_K_M. Existing verified files are reused.")
    parser.add_argument("--repo", help="Hugging Face OWNER/REPOSITORY override")
    parser.add_argument("--profile", choices=("auto", "fast", "official"), default="auto",
                        help="Explicit fast/official can download for another PC without probing this GPU")
    parser.add_argument("--gpu", help="NVIDIA GPU index or full GPU UUID for automatic selection")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help="Alternate model manifest (default: models/manifest.json)")
    parser.add_argument("--timeout", type=float, default=60, help="Per-read HTTPS timeout in seconds (60)")
    parser.add_argument("--retries", type=int, default=3, help="Retries after a transient failure (3)")
    parser.add_argument("--pause-on-exit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.retries < 0 or args.retries > 10:
        parser.error("timeout must be positive; retries must be between 0 and 10")
    root = args.root.resolve()
    manifest = load_manifest(args.manifest.resolve() if args.manifest else root/"models/manifest.json")
    profile, reason = select_profile(root, args.profile, args.gpu)
    item = manifest["files"][profile]
    url = model_url(args.repo or manifest.get("repo_id"), manifest.get("revision", "main"), item["filename"])
    print(f"Model profile: {profile}; {reason}")
    print(f"Source: {args.repo or manifest.get('repo_id')}; revision: {manifest.get('revision', 'main')}")
    destination = root/"models"/item["filename"]
    result = install_model(url, destination, item["size_bytes"], item["sha256"],
                           retries=args.retries, timeout=args.timeout, token=os.environ.get("HF_TOKEN"))
    print(f"Ready ({result}): {destination}\nNext: translate-batch.cmd examples\\input.jsonl output.jsonl\nOr: start-server.cmd")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        result = main()
    except (ConfigError, SetupError, OSError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        result = 1
    except KeyboardInterrupt:
        print("\nDownload cancelled; partial data retained for the next run.", file=sys.stderr)
        result = 130
    if "--pause-on-exit" in sys.argv and sys.stdin.isatty():
        try:
            input("Press Enter to close...")
        except (EOFError, KeyboardInterrupt):
            pass
    raise SystemExit(result)
