"""Maintainer login and allowlisted Hugging Face model publication."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
# Keep credentials out of Git and out of runtime release payloads.
os.environ['HF_HOME'] = str(ROOT/'.local/huggingface')
os.environ['HF_TOKEN_PATH'] = str(ROOT/'.local/huggingface/token')
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
for inherited_key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_OIDC_RESOURCE'):
    os.environ.pop(inherited_key, None)


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_remote_models(info, manifest):
    remote = {entry.rfilename: entry for entry in info.siblings}
    verified = {}
    for item in manifest['files'].values():
        name = item['filename']
        entry = remote.get(name)
        if (entry is None or entry.size != item['size_bytes'] or not entry.lfs or
                entry.lfs.sha256 != item['sha256'] or entry.lfs.size != item['size_bytes']):
            raise RuntimeError(f'Remote model verification failed: {name}')
        verified[name] = {'size_bytes': entry.size, 'sha256': entry.lfs.sha256}
    return verified


def remote_file_state(info):
    return {
        entry.rfilename: {
            'size_bytes': entry.size, 'git_blob_sha1': entry.blob_id,
            'lfs_sha256': entry.lfs.sha256 if entry.lfs else None,
        }
        for entry in info.siblings
    }


def update_card(api, commit_operation_add, repo, manifest, private):
    # This mode must never create a repository or construct a model upload operation.
    before = api.model_info(repo, revision='main', files_metadata=True)
    if before.private != private:
        raise RuntimeError('Existing repository visibility differs from the requested setting; refusing to update it')
    before_models = verify_remote_models(before, manifest)
    sources = (
        ('huggingface/README.md', 'README.md'),
        ('huggingface/MODEL_CHANGES.md', 'MODEL_CHANGES.md'),
    )
    operations = []
    updated_files = {}
    for source, target in sources:
        path = ROOT/source
        if not path.is_file():
            raise FileNotFoundError(path)
        # Freeze these small documents so the committed bytes match the recorded hashes.
        content = path.read_bytes()
        operations.append(commit_operation_add(path_in_repo=target, path_or_fileobj=content))
        updated_files[target] = {
            'size_bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(content)).encode('ascii') + b'\0' + content).hexdigest(),
        }
    print(f'Updating README.md and MODEL_CHANGES.md in {repo} ...', flush=True)
    commit = api.create_commit(
        repo_id=repo, repo_type='model', revision='main', parent_commit=before.sha,
        operations=operations, commit_message='Update Windows project links and measured translation performance',
    )
    report = {
        'ok': False, 'operation': 'update-card', 'status': 'submitted; verification pending',
        'repo_id': repo, 'url': f'https://huggingface.co/{repo}',
        'before_revision': before.sha, 'revision': commit.oid, 'private': before.private,
        'updated_files': updated_files, 'remote_models_before': before_models,
    }
    report_path = ROOT/'.local/huggingface-card-update.json'
    report_path.parent.mkdir(exist_ok=True)
    def save_report():
        report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    save_report()
    try:
        after = api.model_info(repo, revision=commit.oid, files_metadata=True)
        if after.private != private or after.sha != commit.oid:
            raise RuntimeError('Repository visibility or committed revision verification failed')
        after_models = verify_remote_models(after, manifest)
        before_files, after_files = remote_file_state(before), remote_file_state(after)
        unchanged_before = {name: item for name, item in before_files.items() if name not in updated_files}
        unchanged_after = {name: item for name, item in after_files.items() if name not in updated_files}
        if unchanged_before != unchanged_after:
            raise RuntimeError('A file outside README.md and MODEL_CHANGES.md changed; inspect the committed revision')
        for name, expected in updated_files.items():
            actual = after_files.get(name)
            if (actual is None or actual['size_bytes'] != expected['size_bytes'] or
                    actual['git_blob_sha1'] != expected['git_blob_sha1']):
                raise RuntimeError(f'Updated document verification failed: {name}')
        report.update(ok=True, status='verified', remote_models_after=after_models,
                      remote_models_verified_before_and_after=True, non_card_files_unchanged=True)
    except Exception as error:
        report.update(status='post-commit verification failed', error_type=type(error).__name__)
        save_report()
        raise
    save_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--login', action='store_true', help='Enter a write token locally; never pass a token as an argument')
    mode.add_argument('--login-stdin', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--upload', action='store_true', help='Upload the verified models and allowlisted accompanying files')
    mode.add_argument('--update-card', action='store_true', help='Update only README.md and MODEL_CHANGES.md in an existing repository; verify remote model hashes before and after')
    parser.add_argument('--private', action='store_true', help='Create a private repository (default: public)')
    parser.add_argument('--repo', help='Default: models/manifest.json repo_id')
    args = parser.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd, get_token, login, set_client_factory
    import httpx
    set_client_factory(lambda: httpx.Client(follow_redirects=True,
        timeout=httpx.Timeout(60, connect=15, write=600, pool=15)))
    if args.login or args.login_stdin:
        if args.login and not sys.stdin.isatty():
            raise RuntimeError('Run huggingface-login.cmd in a local interactive terminal. Token input must be hidden.')
        if args.login:
            print('Create a write token at https://huggingface.co/settings/tokens')
            print('Paste it below. Input is hidden and stored only in ignored .local/huggingface/.')
        token = (sys.stdin.readline() if args.login_stdin else getpass.getpass('Hugging Face write token: ')).strip()
        if not re.fullmatch(r'hf_[A-Za-z0-9_-]{10,1000}', token):
            raise ValueError('Token format invalid. Paste only the complete hf_ token, without quotes, spaces or extra lines.')
        identity = HfApi(token=token).whoami()
        login(token=token, add_to_git_credential=False)
        print(f"Logged in as {identity['name']}. You may close this window.")
        return 0
    token = get_token()
    if not token:
        raise RuntimeError('No local Hugging Face login. Run huggingface-login.cmd first; do not send tokens in chat.')
    api = HfApi(token=token)
    print('Checking Hugging Face account...', flush=True)
    identity = api.whoami()
    if args.status or not (args.upload or args.update_card):
        print(json.dumps({'logged_in': True, 'username': identity['name']}, ensure_ascii=False))
        return 0
    manifest = json.loads((ROOT/'models/manifest.json').read_text(encoding='utf-8-sig'))
    repo = args.repo or manifest['repo_id']
    if not repo or repo.count('/') != 1:
        raise ValueError('Specify the exact owner/model repository')
    if args.update_card:
        return update_card(api, CommitOperationAdd, repo, manifest, args.private)
    operations = []
    for item in manifest['files'].values():
        name = item['filename']
        if Path(name).name != name:
            raise ValueError('Invalid model filename')
        path = ROOT/'models'/name
        print(f'Checking {name} ...', flush=True)
        if path.stat().st_size != item['size_bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Model checksum mismatch: {name}')
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
    for source, target in (
        ('huggingface/README.md', 'README.md'),
        ('huggingface/MODEL_CHANGES.md', 'MODEL_CHANGES.md'),
        ('licenses/model-Hy-MT2.txt', 'LICENSE'),
        ('models/manifest.json', 'manifest.json'),
        ('huggingface/provenance.json', 'provenance.json'),
    ):
        path = ROOT/source
        if not path.is_file():
            raise FileNotFoundError(path)
        operations.append(CommitOperationAdd(path_in_repo=target, path_or_fileobj=str(path)))
    repo_url = api.create_repo(repo_id=repo, repo_type='model', private=args.private, exist_ok=True)
    info = api.model_info(repo, files_metadata=True)
    if info.private != args.private:
        raise RuntimeError('Existing repository visibility differs from the requested setting; refusing to change it silently')
    print(f'Uploading verified files to {repo_url} ...', flush=True)
    commit = api.create_commit(repo_id=repo, repo_type='model', operations=operations,
                               commit_message='Add verified Hy-MT2 NVFP4 and Q4_K_M fused GGUF models')
    info = api.model_info(repo, revision=commit.oid, files_metadata=True)
    remote = {entry.rfilename: entry for entry in info.siblings}
    for item in manifest['files'].values():
        entry = remote[item['filename']]
        if entry.size != item['size_bytes'] or not entry.lfs or entry.lfs.sha256 != item['sha256']:
            raise RuntimeError(f"Remote model verification failed: {item['filename']}")
    report = {'ok': True, 'repo_id': repo, 'url': str(repo_url), 'revision': commit.oid,
              'private': info.private, 'remote_models_verified': True}
    (ROOT/'.local').mkdir(exist_ok=True)
    (ROOT/'.local/huggingface-upload.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # SDK exceptions must not echo authentication request bodies.
        if type(error).__module__.startswith(('huggingface_hub', 'httpx', 'httpcore')):
            detail = 'Remote request failed. Check local login, repository access and network connectivity.'
        else:
            detail = re.sub(r'hf_[A-Za-z0-9_-]+', '[redacted token]', str(error))
        print(f'Error ({type(error).__name__}): {detail}', file=sys.stderr)
        raise SystemExit(1)
