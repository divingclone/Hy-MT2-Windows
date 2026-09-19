"""Apply a portable ZIP after Tauri's official updater verified its signature.

Runs from a copied Python runtime outside the replaced payload. Only the two
application paths (EXE, payload) are replaced; data and portable marker stay.
Each replacement is journaled and rolled back if another replacement fails.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

EXE = 'hymt-desktop.exe'


def confined_remove(path: Path, parent: Path):
    if path.resolve().parent != parent.resolve() or path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise ValueError('Unsafe update cleanup target')
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def validate_entries(archive: zipfile.ZipFile):
    seen = set()
    total = 0
    for item in archive.infolist():
        name = item.orig_filename
        path = PurePosixPath(name)
        if (not name or '\\' in name or ':' in name or path.is_absolute() or
                any(part in ('.', '..') or part.endswith(('.', ' ')) for part in name.rstrip('/').split('/')) or
                path.parts[0] not in (EXE, 'payload', 'portable.json', 'README-desktop.txt') or
                ((item.external_attr >> 16) & 0o170000) == 0o120000):
            raise ValueError(f'Unsafe update entry: {name}')
        key = name.rstrip('/').casefold()
        if key in seen:
            raise ValueError('Duplicate update path')
        seen.add(key)
        total += item.file_size
        if total > 8 * 1024**3:
            raise ValueError('Portable update exceeds 8 GiB unpacked limit')
    if EXE not in seen or 'payload/scripts/desktop_bridge.py' not in seen:
        raise ValueError('Incomplete portable update')
    return total


def prepare(archive: Path, install: Path, staging: Path):
    install, staging = install.resolve(), staging.resolve()
    if not (install / 'portable.json').is_file() or not staging.is_relative_to(install / 'data'):
        raise ValueError('Update target is not a portable installation')
    next_dir = staging / 'next'
    with zipfile.ZipFile(archive) as zipped:
        total = validate_entries(zipped)
        if shutil.disk_usage(staging).free < total + 300 * 1024**2:
            raise OSError('更新所需磁盘空间不足。')
        confined_remove(next_dir, staging)
        next_dir.mkdir()
        zipped.extractall(next_dir)
    runtime = staging / 'helper-python'
    confined_remove(runtime, staging)
    shutil.copytree(install / 'payload/runtime/python', runtime)
    shutil.copyfile(__file__, staging / 'desktop_update.py')
    # Verify the staged runtime and copied helper before exiting the GUI.
    subprocess.run([str(runtime / 'python.exe'), '-E', '-s', '-c', 'import ctypes,zipfile,json,shutil'], check=True, timeout=20,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def wait_parent(pid: int):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if handle:
        try:
            if kernel.WaitForSingleObject(handle, 120000) != 0:
                raise TimeoutError('HyMT did not exit; update cancelled')
        finally:
            kernel.CloseHandle(handle)


def rename_retry(source: Path, target: Path):
    for attempt in range(40):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.25)


def apply(install: Path, staging: Path, pid: int):
    install, staging = install.resolve(), staging.resolve()
    if not (install / 'portable.json').is_file() or staging != install / 'data/updates':
        raise ValueError('Invalid portable update location')
    wait_parent(pid)
    backup = staging / 'previous'
    # Leave a previous rollback directory for manual recovery if necessary.
    if backup.exists():
        confined_remove(backup, staging)
    backup.mkdir()
    moved, applied = [], []
    try:
        for name in (EXE, 'payload'):
            old, new = install / name, staging / 'next' / name
            if old.is_symlink() or (hasattr(old, 'is_junction') and old.is_junction()):
                raise ValueError('Linked application path is not supported')
            rename_retry(old, backup / name)
            moved.append(name)
            rename_retry(new, old)
            applied.append(name)
    except Exception:
        for name in reversed(applied):
            rename_retry(install / name, staging / 'next' / name)
        for name in reversed(moved):
            rename_retry(backup / name, install / name)
        raise
    (staging / 'result.json').write_text(json.dumps({'ok': True}), encoding='utf-8')
    subprocess.Popen([str(install / EXE)], cwd=install, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'prepare':
            prepare(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        elif sys.argv[1] == 'apply':
            apply(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
        else:
            raise ValueError('Unknown operation')
    except Exception as exc:
        if len(sys.argv) > 3 and sys.argv[1] == 'apply':
            staging = Path(sys.argv[3])
            (staging / 'result.json').write_text(json.dumps({'ok': False, 'error': str(exc)}), encoding='utf-8')
            ctypes.windll.user32.MessageBoxW(None, f'免安装版更新失败，已尝试恢复旧版。\n{exc}\n详情：{staging}', 'HyMT 更新', 0x10)
        raise
