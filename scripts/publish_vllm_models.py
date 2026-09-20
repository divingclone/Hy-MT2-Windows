"""Publish raw safetensors checkpoints to separate verified Hugging Face repositories."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.local/huggingface')
os.environ['HF_TOKEN_PATH'] = str(ROOT / '.local/huggingface/token')
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')
for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_OIDC_RESOURCE'):
    os.environ.pop(name, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', default='divingclone')
    parser.add_argument('--login-stdin', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd, get_token, login, set_client_factory
    import httpx
    set_client_factory(lambda: httpx.Client(follow_redirects=True,
        timeout=httpx.Timeout(60, connect=15, write=600, pool=15)))
    if args.login_stdin:
        token = sys.stdin.readline().strip()
        if not re.fullmatch(r'hf_[A-Za-z0-9_-]{10,1000}', token):
            raise ValueError('Invalid Hugging Face token format')
        try:
            identity = HfApi(token=token).whoami()
            login(token=token, add_to_git_credential=False)
        except Exception:
            # Never copy provider exception details containing credentials into the GUI.
            raise RuntimeError('Hugging Face login failed; check the token and connection') from None
        print(f"Logged in as {identity['name']}. You may close this window.")
        return 0
    token = get_token()
    if not token:
        raise RuntimeError('Local Hugging Face login required')
    api = HfApi(endpoint='https://huggingface.co', token=token)
    anonymous = HfApi(endpoint='https://huggingface.co', token=False)
    if args.owner != api.whoami()['name']:
        raise ValueError('Repository must belong to the authenticated account')
    manifest_path = ROOT / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    report_path = ROOT / 'results/huggingface-raw-upload.json'
    report_path.parent.mkdir(exist_ok=True)
    report = {'ok':False, 'format':'raw checkpoint', 'repositories':{}}
    for profile, suffix in (('fast','NVFP4'),('compat','INT4')):
        repo = args.owner+'/Hy-MT2-1.8B-'+suffix+'-vLLM'
        spec = manifest if profile=='fast' else manifest['checkpoints'][profile]
        sources = {}
        for name, expected in spec['checkpoint_files'].items():
            if Path(name).name != name: raise ValueError('Invalid checkpoint filename')
            path = ROOT/'models'/spec['checkpoint_dir']/name
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size != expected['size_bytes'] or digest != expected['sha256']:
                raise ValueError('Checkpoint digest mismatch: '+name)
            sources[name] = path
        card=(ROOT/'huggingface/vllm/README.md').read_text('utf-8').replace('{VARIANT}',suffix).replace('{REPO}',repo)
        small = {'README.md':card.encode(), 'LICENSE':(ROOT/'licenses/model-Hy-MT2.txt').read_bytes(),
                 'MODEL_CHANGES.md':(ROOT/'huggingface/vllm/MODEL_CHANGES.md').read_bytes(),
                 'checksums.json':(json.dumps(spec['checkpoint_files'],indent=2)+'\n').encode()}
        if profile=='compat': small['quantization.json']=(ROOT/'benchmarks/int4-quantization.json').read_bytes()
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in sources.items()]
        operations += [CommitOperationAdd(path_in_repo=name,path_or_fileobj=data) for name,data in small.items()]
        api.create_repo(repo_id=repo,repo_type='model',private=False,exist_ok=True)
        before=api.model_info(repo,files_metadata=True)
        if before.private: raise ValueError('Existing repository is private')
        print('Publishing raw '+suffix+' checkpoint: '+repo,flush=True)
        commit=api.create_commit(repo_id=repo,repo_type='model',revision='main',parent_commit=before.sha,
            operations=operations,commit_message='Publish calibrated Hy-MT2 '+suffix+' safetensors checkpoint')
        report['repositories'][profile]={'repo_id':repo,'revision':commit.oid,'verified':False}
        report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        info=anonymous.model_info(repo,revision=commit.oid,files_metadata=True)
        if info.private or info.sha != commit.oid: raise RuntimeError('Anonymous revision check failed')
        remote={f.rfilename:f for f in info.siblings}
        for name in [*sources,*small]:
            item=remote[name]
            if name in sources:
                expected=spec['checkpoint_files'][name]
                size,digest=expected['size_bytes'],expected['sha256']
            else:
                size,digest=len(small[name]),hashlib.sha256(small[name]).hexdigest()
            if item.size!=size: raise RuntimeError('Remote size mismatch: '+name)
            if item.lfs:
                if item.lfs.sha256!=digest: raise RuntimeError('Remote SHA256 mismatch: '+name)
            else:
                data=sources[name].read_bytes() if name in sources else small[name]
                blob=hashlib.sha1(b'blob '+str(size).encode()+b'\0'+data).hexdigest()
                if item.blob_id!=blob: raise RuntimeError('Remote git blob mismatch: '+name)
        report['repositories'][profile]['verified']=True
        report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    manifest.update(repo_id=None,revision='main',repositories={
        key:{k:v for k,v in value.items() if k in ('repo_id','revision')}
        for key,value in report['repositories'].items()})
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report.update(ok=True,anonymous_access=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Avoid SDK exceptions exposing request headers or credentials.
        print('Model publication failed: '+type(error).__name__+'. Check local credentials, files and network.', file=sys.stderr)
        raise SystemExit(1)
