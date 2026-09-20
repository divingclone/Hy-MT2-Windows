"""Stage the existing allowlisted runtime and build portable desktop.

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
from desktop_update import ROOT_DLLS
from runtime_filter import excluded

DESKTOP = ROOT / 'desktop'
PAYLOAD = DESKTOP / 'src-tauri/resources/payload'


def stage(include_models=False,include_webview2=False):
    python = ROOT / 'runtime/python'
    if not (python / 'python.exe').exists():
        python /= 'cpython-3.12-windows-x86_64-none'
    entries, problems, _ = make_plan(ROOT, ROOT / 'bin', python, include_models=include_models,include_webview2=include_webview2)
    if problems:
        raise RuntimeError('\n'.join(problems))
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    desired = {entry.relative for entry in entries} | {'scripts/desktop_bridge.py', 'scripts/desktop_update.py', 'licenses/DESKTOP-DEPENDENCIES.txt'}
    desired.add('.gitkeep')
    # Clean only files inside the explicitly resolved generated staging path.
    expected = (DESKTOP / 'src-tauri/resources/payload').resolve()
    if PAYLOAD.resolve() != expected or PAYLOAD.is_symlink():
        raise ValueError('Unsafe staging directory')
    for existing in PAYLOAD.rglob('*'):
        if existing.is_symlink() or (hasattr(existing, 'is_junction') and existing.is_junction()):
            raise ValueError('Staging may not contain links')
        if existing.is_file() and existing.relative_to(PAYLOAD).as_posix() not in desired:
            existing.unlink()
    # Empty package/dist-info directories still look importable to Python.
    # Remove empty directories only, under the verified generated staging root.
    for directory in sorted((p for p in PAYLOAD.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        if not any(directory.iterdir()): directory.rmdir()
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


def package_portable(output: Path, version: str, include_models=False,include_webview2=False):
    if include_webview2 and not (PAYLOAD/'runtime/webview2/msedgewebview2.exe').is_file():
        raise ValueError('Offline WebView2 was not staged; run without --skip-stage')
    folder = output / f'HyMT-{version}-windows-x64-portable'
    if folder.exists():
        raise FileExistsError(f'{folder} already exists; use a fresh output directory')
    folder.mkdir(parents=True)
    shutil.copyfile(DESKTOP / 'src-tauri/target/release/hymt-desktop.exe', folder / 'hymt-desktop.exe')
    def exclude(directory,names):
        skipped={n for n in names if n in ('.gitkeep','__pycache__')}
        skipped.update(n for n in names if excluded((Path(directory)/n).relative_to(PAYLOAD).as_posix(),include_webview2=include_webview2))
        if Path(directory).resolve()==(PAYLOAD/'models').resolve() and not include_models:
            # Also protect --skip-stage against an older staging tree containing weights.
            skipped.update(n for n in names if n != 'manifest.json')
        if Path(directory).parts[-4:]==('tilelang','3rdparty','composable_kernel','library'):
            skipped.add('src')
        return skipped
    shutil.copytree(PAYLOAD, folder / 'payload', ignore=exclude)
    for name in ROOT_DLLS:
        shutil.copy2(PAYLOAD/'runtime/vllm'/name,folder/name)
    (folder / 'portable.json').write_text('{"format":1}\n', encoding='utf-8')
    (folder / 'README-desktop.txt').write_text('HyMT 翻译后端\n解压整个文件夹，然后双击 hymt-desktop.exe。\n模型单独托管在 Hugging Face。首次在模型页按显卡下载 NVFP4 或 INT4，或导入经过验证的模型 ZIP；已有模型自动校验复用。启动服务后复制 Base URL、模型名和 API Key 到其他应用。\n设置和密钥保存在 data 目录；模型优先保存在程序旁的 models 文件夹，可随整个文件夹搬走。退出应用自动停止后台推理。\n需要 NVIDIA RTX 30/40/50 对应受支持架构与驱动 596.36 或更新；RTX 30/40 尚待实卡验证。\nPython、CUDA 运行库、TinyCC、预编译采样内核及 MSVC 运行库已随包提供，无需安装开发工具。界面优先使用系统 WebView2，缺失时自动下载并校验应用本地版本，无需手动安装；缺失运行时的首次启动需联网。\n第三方组件须遵守 payload/licenses 中随附的许可条款，使用前请阅读 WEBVIEW2-FIXED-LICENSE.txt。界面包含 Microsoft Defender SmartScreen，其信息收集与传输按 https://aka.ms/privacy 及 https://learn.microsoft.com/en-us/microsoft-edge/privacy-whitepaper#smartscreen 说明。\n', encoding='utf-8')
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
    parser.add_argument('--skip-stage', action='store_true', help='Reuse an already verified payload when rebuilding only the desktop UI')
    parser.add_argument('--include-models', action='store_true', help='Explicitly include both model ZIPs for an offline package')
    parser.add_argument('--include-webview2', action='store_true', help='Include app-local WebView2 for offline first launch')
    parser.add_argument('--signed', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/desktop')
    args = parser.parse_args()
    if not args.skip_stage:
        stage(include_models=args.include_models,include_webview2=args.include_webview2)
    elif not (PAYLOAD/'runtime/vllm/python.exe').is_file():
        raise ValueError('No staged vLLM payload; run without --skip-stage first')
    if args.stage_only:
        return
    config = json.loads((DESKTOP / 'src-tauri/tauri.conf.json').read_text('utf-8'))
    version = config['version']
    command = ['npm.cmd', 'run', 'tauri', 'build', '--', '--no-bundle']
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
    archive = package_portable(output, version, include_models=args.include_models,include_webview2=args.include_webview2)
    if args.signed:
        sign_env = os.environ.copy()
        key_value = sign_env.get('TAURI_SIGNING_PRIVATE_KEY', '')
        if len(key_value) < 1024 and Path(key_value).is_file():
            sign_env['TAURI_SIGNING_PRIVATE_KEY_PATH'] = key_value
            sign_env.pop('TAURI_SIGNING_PRIVATE_KEY', None)
        subprocess.run(['npm.cmd', 'run', 'tauri', 'signer', 'sign', '--', str(archive)], cwd=DESKTOP, env=sign_env, check=True)
        release = os.environ.get('HYMT_RELEASE_BASE_URL', f'https://github.com/divingclone/Hy-MT2-Windows/releases/download/desktop-v{version}')
        manifest = {'version': version, 'notes': 'HyMT 桌面端更新', 'platforms': {
            'windows-x86_64-portable': {'url': f'{release}/{archive.name}', 'signature': archive.with_suffix('.zip.sig').read_text('utf-8').strip()}}}
        (output / 'desktop-latest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(file)}  {file.name}\n' for file in (archive,)), encoding='utf-8')
    print(f'Portable: {archive}', flush=True)


if __name__ == '__main__':
    main()
