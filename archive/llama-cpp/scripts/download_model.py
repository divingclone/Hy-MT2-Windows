"""Download a pinned HF artifact with range checks and SHA256 verification."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('repo')
    p.add_argument('filename')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    req = urllib.request.Request(f'https://huggingface.co/api/models/{a.repo}?blobs=true')
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                meta = json.load(r)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    f = next(f for f in meta['siblings'] if f['rfilename'] == a.filename)
    size, sha = f['size'], f['lfs']['sha256']
    name = Path(a.filename).name
    output = ROOT / 'models' / name
    parts = ROOT / 'cache' / 'downloads' / sha
    parts.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{a.repo}/resolve/{meta["sha"]}/{a.filename}?download=true'
    block = 8 * 1024 * 1024
    chunks = [(i, min(i + block, size) - 1) for i in range(0, size, block)]
    started = time.monotonic()

    def chunk(bounds):
        start, end = bounds
        path = parts / str(start)
        count = end - start + 1
        if path.exists() and path.stat().st_size == count:
            return count
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=60) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Invalid range response')
                    with path.open('wb') as out:
                        remaining = count
                        while remaining:
                            buf = r.read(min(1024 * 1024, remaining))
                            if not buf:
                                raise EOFError('Truncated download')
                            out.write(buf)
                            remaining -= len(buf)
                return count
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(chunk, b) for b in chunks]):
            done += future.result()
            elapsed = time.monotonic() - started
            print(f'{name}: {done / size:.1%}, {done / 1048576 / max(elapsed, .01):.1f} MiB/s', flush=True)
    temp = output.with_suffix(output.suffix + '.verified-part')
    digest = hashlib.sha256()
    with temp.open('wb') as out:
        for start, _ in chunks:
            with (parts / str(start)).open('rb') as part:
                while buf := part.read(1024 * 1024):
                    digest.update(buf)
                    out.write(buf)
    if digest.hexdigest() != sha:
        raise RuntimeError('SHA256 mismatch; original output was not changed')
    temp.replace(output)
    manifest = {'repository': a.repo, 'revision': meta['sha'], 'filename': name,
                'size': size, 'sha256': sha, 'verified': True, 'url': url}
    output.with_suffix('.download.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
