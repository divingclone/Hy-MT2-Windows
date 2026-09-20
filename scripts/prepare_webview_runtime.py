"""Fetch and unpack the pinned Microsoft Fixed Version WebView2, without installing it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request
import time
import sys
import shutil

VERSION='153.0.4234.48'
URL='https://msedge.sf.dl.delivery.mp.microsoft.com/filestreamingservice/files/08cd33ee-d109-49b8-9301-9f0bea43c575/Microsoft.WebView2.FixedVersionRuntime.153.0.4234.48.x64.cab'
SHA256='11e8240cb0bc56dcd3e4498907203c251346f65107fe35a3a13e152c7d51c79e'
SIZE=308509880


def ready(target):
    try:
        info=json.loads((target/'hymt-webview2.json').read_text(encoding='utf-8'))
        return info['cab_sha256']==SHA256 and (target/'msedgewebview2.exe').is_file() and (target/'msedge.dll').is_file()
    except (OSError, ValueError, KeyError):
        return False


def download(cab, progress):
    start=cab.stat().st_size if cab.exists() else 0
    if start>SIZE:
        raise ValueError('WebView2 partial download exceeds expected size')
    if start==SIZE:
        return
    request=urllib.request.Request(URL, headers={'Range':f'bytes={start}-'} if start else {})
    with urllib.request.urlopen(request,timeout=30) as response:
        if start and response.status==206:
            if not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                raise ValueError('Invalid WebView2 download range')
            mode='ab'
        else:
            start=0;mode='wb'
        with cab.open(mode) as output:
            while chunk:=response.read(1024*1024):
                start+=len(chunk)
                if start>SIZE: raise ValueError('WebView2 download exceeds expected size')
                output.write(chunk)
                progress(f'正在下载界面组件：{start/1024**2:.0f} / {SIZE/1024**2:.0f} MiB')
    if start!=SIZE: raise ValueError('WebView2 download incomplete; restart HyMT to resume')


def ensure_runtime(target, cab=None, progress=lambda message: None):
    """Publish only a hash/signature-verified runtime; interrupted CABs resume."""
    import msvcrt
    target=Path(target).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    # OS releases this lock even after forced termination. Other launches wait.
    with target.with_name(target.name+'.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0');lock.flush()
        while True:
            try:
                lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                progress('正在等待另一窗口完成界面组件下载…');time.sleep(.2)
        prefix=f'webview-build-{VERSION}-'
        # A closed bootstrap Job may interrupt expand.exe. Clean only this
        # version's generated staging directories, under its exclusive lock.
        for stale in target.parent.glob(prefix+'*'):
            if stale.is_symlink() or stale.is_junction() or stale.resolve().parent!=target.parent:
                raise ValueError('Unsafe WebView2 staging directory')
            if stale.is_dir(): shutil.rmtree(stale)
        partial=target.with_name(target.name+'.cab.part')
        if ready(target):
            if partial.is_file(): partial.unlink()
            return target
        if target.exists(): raise ValueError(f'Incomplete WebView2 directory: {target}')
        cached=cab is None
        cab=Path(cab).resolve() if cab else partial
        if cached: download(cab,progress)
        progress('正在校验界面组件…')
        with cab.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=SHA256:
            # Our own corrupt partial file must not poison every future retry.
            if cached: cab.unlink()
            raise ValueError('WebView2 CAB digest mismatch')
        with tempfile.TemporaryDirectory(prefix=prefix,dir=target.parent) as temp:
            staging=Path(temp).resolve()
            progress('正在准备界面组件，请稍候…')
            extracted=staging/'extracted';extracted.mkdir()
            subprocess.run(['expand.exe',str(cab),'-F:*',str(extracted)],stdout=subprocess.DEVNULL,
                           check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            source=extracted/f'Microsoft.WebView2.FixedVersionRuntime.{VERSION}.x64'
            env=os.environ.copy();env['HYMT_VERIFY_FILE']=str(source/'msedgewebview2.exe')
            powershell=Path(env.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0'
            # PowerShell 7 hosts can pass an incompatible module search path to
            # Python; load the Windows signature cmdlet from its system module.
            env['PSModulePath']=str(powershell/'Modules')
            signature=json.loads(subprocess.check_output([str(powershell/'powershell.exe'),'-NoProfile','-Command',
                '[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); $ErrorActionPreference="Stop"; $s=Get-AuthenticodeSignature -LiteralPath $env:HYMT_VERIFY_FILE; @{status=$s.Status.ToString();signer=$s.SignerCertificate.Subject}|ConvertTo-Json -Compress'],
                env=env,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW))
            if signature['status']!='Valid' or 'CN=Microsoft Corporation,' not in signature['signer']:
                raise ValueError('WebView2 Microsoft signature verification failed')
            if not source.resolve().is_relative_to(staging) or target.parent!=staging.parent:
                raise ValueError('Unsafe runtime staging path')
            (source/'hymt-webview2.json').write_text(json.dumps({'version':VERSION,'source':URL,'cab_sha256':SHA256,
                'signature':signature,'distribution':'Fixed Version; app-local, no system installation'},indent=2),encoding='utf-8')
            source.rename(target)
        if cached: cab.unlink()
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cab',type=Path,help='Use an already downloaded CAB with the pinned digest')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'runtime/webview2')
    parser.add_argument('--cache',type=Path,help='Versioned fallback runtime cache')
    parser.add_argument('--progress',action='store_true',help='Show native progress without requiring WebView2')
    parser.add_argument('--wait-for-parent',action='store_true')
    args=parser.parse_args()
    if args.wait_for_parent and sys.stdin.readline().strip()!='go': raise SystemExit(1)
    target=args.cache/VERSION if args.cache else args.output
    if args.progress and not ready(target):
        from webview_progress import run_with_progress
        result=run_with_progress(lambda update: ensure_runtime(target,args.cab,update))
    else:
        result=ensure_runtime(target,args.cab)
    print(result,flush=True)


if __name__=='__main__':main()
