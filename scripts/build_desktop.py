"""Stage the existing allowlisted runtime and build NSIS + portable desktop.

No signing private key is generated or written by this script. Release signing
uses TAURI_SIGNING_PRIVATE_KEY and HYMT_UPDATER_PUBLIC_KEY supplied by the owner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from package_windows import ROOT, make_plan, verify_python, sha256

DESKTOP = ROOT / 'desktop'
PAYLOAD = DESKTOP / 'src-tauri/resources/payload'


def stage():
    python = ROOT / 'runtime/python'
    if not (python / 'python.exe').exists():
        python /= 'cpython-3.12-windows-x86_64-none'
    entries, problems, _ = make_plan(ROOT, ROOT / 'bin', python)
    if problems:
        raise RuntimeError('\n'.join(problems))
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    desired = {entry.relative for entry in entries} | {'scripts/desktop_bridge.py', 'scripts/desktop_update.py', 'licenses/DESKTOP-DEPENDENCIES.txt'}
    # Clean only files inside the explicitly resolved generated staging path.
    expected = (DESKTOP / 'src-tauri/resources/payload').resolve()
    if PAYLOAD.resolve() != expected or PAYLOAD.is_symlink():
        raise ValueError('Unsafe staging directory')
    for existing in PAYLOAD.rglob('*'):
        if existing.is_symlink() or (hasattr(existing, 'is_junction') and existing.is_junction()):
            raise ValueError('Staging may not contain links')
        if existing.is_file() and existing.relative_to(PAYLOAD).as_posix() not in desired | {'.gitkeep'}:
            existing.unlink()
    for source, relative in [(entry.source, entry.relative) for entry in entries] + [(ROOT / 'scripts' / name, 'scripts/' + name) for name in ('desktop_bridge.py', 'desktop_update.py')]:
        target = PAYLOAD / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or source.stat().st_size != target.stat().st_size or sha256(source) != sha256(target):
            shutil.copyfile(source, target)
    licenses = ['HyMT desktop: Tauri 2 + Svelte\nSee the adjacent project and inference licenses.\n']
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--locked', '--filter-platform', 'x86_64-pc-windows-msvc', '--format-version', '1', '--manifest-path', str(DESKTOP / 'src-tauri/Cargo.toml')], text=True, encoding='utf-8'))
    for package in sorted(metadata['packages'], key=lambda item: item['name']):
        licenses.append(f"\n{package['name']} {package['version']} | {package.get('license')} | {package.get('repository')}\n")
        directory = Path(package['manifest_path']).parent
        for file in sorted(directory.glob('*')):
            if file.is_file() and file.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
                licenses.append(file.read_text('utf-8', errors='replace') + '\n')
    for name in ('svelte', '@tauri-apps/api'):
        directory = DESKTOP / 'node_modules' / name
        for file in directory.glob('LICENSE*'):
            licenses.append(f'\n{name}\n' + file.read_text('utf-8', errors='replace'))
    (PAYLOAD / 'licenses/DESKTOP-DEPENDENCIES.txt').write_text('\n'.join(licenses), encoding='utf-8')
    verify_python(PAYLOAD)
    print(f'Staged {len(entries)} runtime files plus desktop workers.', flush=True)


def package_portable(output: Path, version: str):
    folder = output / f'HyMT-{version}-windows-x64-portable'
    if folder.exists():
        raise FileExistsError(f'{folder} already exists; use a fresh output directory')
    folder.mkdir(parents=True)
    shutil.copyfile(DESKTOP / 'src-tauri/target/release/hymt-desktop.exe', folder / 'hymt-desktop.exe')
    shutil.copytree(PAYLOAD, folder / 'payload', ignore=shutil.ignore_patterns('.gitkeep', '__pycache__'))
    (folder / 'portable.json').write_text('{"format":1}\n', encoding='utf-8')
    (folder / 'README-desktop.txt').write_text('HyMT 翻译后端\n解压整个文件夹，然后双击 hymt-desktop.exe。\n首次在模型页下载或导入模型，启动服务后复制 Base URL、模型名和 API Key 到其他应用。\n设置和密钥保存在 data 目录；模型优先保存在程序旁的 models 文件夹，已有模型会校验后自动复用。搬走整个文件夹即可继续使用。退出应用自动停止后台推理。\n需要 NVIDIA 驱动 580.88 或更新，以及 Microsoft Edge WebView2 Runtime。\n如果没有 WebView2，可使用安装版自动安装。\n', encoding='utf-8')
    archive = output / (folder.name + '.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as zipped:
        for file in sorted(folder.rglob('*')):
            if file.is_file():
                zipped.write(file, file.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip():
            raise ValueError('Portable ZIP verification failed')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--signed', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/desktop')
    args = parser.parse_args()
    stage()
    if args.stage_only:
        return
    config = json.loads((DESKTOP / 'src-tauri/tauri.conf.json').read_text('utf-8'))
    version = config['version']
    command = ['npm.cmd', 'run', 'tauri', 'build', '--', '--bundles', 'nsis']
    if args.signed:
        public = os.environ.get('HYMT_UPDATER_PUBLIC_KEY', '').strip()
        if not public or not os.environ.get('TAURI_SIGNING_PRIVATE_KEY'):
            raise ValueError('Signed build requires HYMT_UPDATER_PUBLIC_KEY and TAURI_SIGNING_PRIVATE_KEY')
        overlay = {'bundle': {'createUpdaterArtifacts': True}, 'plugins': {'updater': {
            'pubkey': public, 'endpoints': [os.environ.get('HYMT_UPDATE_URL', 'https://github.com/divingclone/Hy-MT2-Windows/releases/latest/download/desktop-latest.json')]}}}
        overlay_path = DESKTOP / 'src-tauri/tauri.release.conf.json'
        overlay_path.write_text(json.dumps(overlay), encoding='utf-8')
        command += ['--config', str(overlay_path)]
    subprocess.run(command, cwd=DESKTOP, check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = package_portable(output, version)
    installers = list((DESKTOP / 'src-tauri/target/release/bundle/nsis').glob(f'HyMT_{version}_x64-setup.exe'))
    if len(installers) != 1:
        raise ValueError('Expected exactly one NSIS installer')
    installer = output / installers[0].name
    shutil.copyfile(installers[0], installer)
    if args.signed:
        sign_env = os.environ.copy()
        key_value = sign_env.get('TAURI_SIGNING_PRIVATE_KEY', '')
        if len(key_value) < 1024 and Path(key_value).is_file():
            sign_env['TAURI_SIGNING_PRIVATE_KEY_PATH'] = key_value
            sign_env.pop('TAURI_SIGNING_PRIVATE_KEY', None)
        subprocess.run(['npm.cmd', 'run', 'tauri', 'signer', 'sign', '--', str(archive)], cwd=DESKTOP, env=sign_env, check=True)
        signature = installers[0].with_suffix('.exe.sig')
        shutil.copyfile(signature, installer.with_suffix('.exe.sig'))
        release = os.environ.get('HYMT_RELEASE_BASE_URL', f'https://github.com/divingclone/Hy-MT2-Windows/releases/download/desktop-v{version}')
        manifest = {'version': version, 'notes': 'HyMT 桌面端更新', 'platforms': {
            'windows-x86_64': {'url': f'{release}/{installer.name}', 'signature': signature.read_text('utf-8').strip()},
            'windows-x86_64-portable': {'url': f'{release}/{archive.name}', 'signature': archive.with_suffix('.zip.sig').read_text('utf-8').strip()}}}
        (output / 'desktop-latest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(file)}  {file.name}\n' for file in (archive, installer)), encoding='utf-8')
    print(f'Installer: {installer}\nPortable: {archive}', flush=True)


if __name__ == '__main__':
    main()
