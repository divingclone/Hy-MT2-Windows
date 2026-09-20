"""Fetch untouched upstream release artifacts and Tencent's official Q4_K_M."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TAG='b10964'  # Official stable v0.4.1's nightly-tag.txt, resolved 2026-09-20.
REVISION='a0c709d9fac510f2c807aa3af52872340dc37a4a'
MODEL='Hy-MT2-1.8B-Q4_K_M.gguf'
MODEL_SHA='dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    dest=ROOT/'.local/official-api-baseline';dest.mkdir(parents=True,exist_ok=True)
    request=urllib.request.Request(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}',headers={'User-Agent':'HyMT-benchmark'})
    with urllib.request.urlopen(request,timeout=60) as response:release=json.load(response)
    assets=[a for a in release['assets'] if a['name'] in (
        f'llama-{TAG}-bin-win-cuda-13.3-x64.zip','cudart-llama-bin-win-cuda-13.3-x64.zip')]
    assert len(assets)==2
    records=[{'filename':a['name'],'url':a['browser_download_url'],'bytes':a['size'],'sha256':a['digest'].removeprefix('sha256:')} for a in assets]
    records.append({'filename':MODEL,'url':f'https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/{REVISION}/{MODEL}',
                    'bytes':1133080448,'sha256':MODEL_SHA})
    def fetch(record):
        target=dest/record['filename']
        if not target.is_file():
            partial=target.with_name(target.name+'.part')
            offset=partial.stat().st_size if partial.exists() else 0
            request=urllib.request.Request(record['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=120) as response:
                append=offset>0 and response.status==206
                if append:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                with partial.open('ab' if append else 'wb') as output:
                    while chunk:=response.read(4*1024**2):output.write(chunk)
            assert partial.stat().st_size==record['bytes'] and digest(partial)==record['sha256']
            partial.rename(target)
        assert target.stat().st_size==record['bytes'] and digest(target)==record['sha256']
        print('Verified '+record['filename'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,records))
    binary=dest/'bin';binary.mkdir(exist_ok=True)
    inventory={}
    for record in records[:2]:
        with zipfile.ZipFile(dest/record['filename']) as archive:
            for entry in archive.infolist():
                path=binary/entry.filename
                if not path.resolve().is_relative_to(binary.resolve()):raise ValueError('Unsafe archive entry')
                if entry.is_dir():continue
                data=archive.read(entry)
                if path.exists():assert path.read_bytes()==data
                else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                inventory[entry.filename]=hashlib.sha256(data).hexdigest()
    provenance={'official_release':'v0.4.1','binary_tag':TAG,'release_url':release['html_url'],
        'model_repo':'tencent/Hy-MT2-1.8B-GGUF','model_revision':REVISION,'downloads':records,
        'unmodified_extracted_files':inventory}
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(binary,flush=True)


if __name__=='__main__':main()
