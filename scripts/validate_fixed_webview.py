"""Start an owned portable desktop and verify system/bundled/downloaded WebView2.

Run only after packaging has completed: launching the app creates user data.
The test stops only its own process tree and never closes an existing desktop.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--mode',choices=('system','fallback','bundled'),default='system')
    args=p.parse_args()
    folder=args.directory.resolve();report=args.report.resolve()
    if report.exists() or report.is_relative_to(folder): raise ValueError('Choose a new external report')
    if any(proc.info['name'].lower()=='hymt-desktop.exe' for proc in psutil.process_iter(['name'])):
        raise RuntimeError('An existing desktop is running; this test will not interrupt it')
    from prepare_webview_runtime import VERSION
    expected=(folder/'data/webview-runtime'/VERSION/'msedgewebview2.exe') if args.mode=='fallback' else folder/'payload/runtime/webview2/msedgewebview2.exe'
    if args.mode=='bundled' and not expected.is_file(): raise ValueError('Fixed WebView2 missing')
    # Exclude host SDK/CRT search paths, while retaining Windows system DLLs.
    env={k:v for k,v in os.environ.items() if not k.startswith(('CUDA','VC','VS','PYTHON','WEBVIEW2'))}
    env['PATH']=str(Path(env.get('SystemRoot','C:/Windows'))/'System32')
    env.pop('HYMT_WEBVIEW2_FORCE_FALLBACK',None)
    if args.mode!='system': env['HYMT_WEBVIEW2_FORCE_FALLBACK']='1'
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    process=subprocess.Popen([str(folder/'hymt-desktop.exe')],cwd=folder,env=env,startupinfo=startup)
    result={'passed':False,'pid':process.pid,'mode':args.mode,'expected_browser':str(expected) if args.mode!='system' else 'registered Evergreen'}
    try:
        deadline=time.monotonic()+300;browser_paths=[]
        while process.poll() is None and time.monotonic()<deadline:
            children=psutil.Process(process.pid).children(recursive=True)
            browser_paths=[p.exe() for p in children if p.name().lower()=='msedgewebview2.exe']
            if browser_paths: break
            time.sleep(.5)
        valid=bool(browser_paths) and all(
            (not Path(p).resolve().is_relative_to(folder) and 'edgewebview' in p.lower()) if args.mode=='system'
            else Path(p).resolve()==expected for p in browser_paths)
        if not valid: raise RuntimeError(f'Desktop used an unexpected runtime: {browser_paths}')
        time.sleep(5)
        if process.poll() is not None: raise RuntimeError('Desktop exited during startup')
        dlls=sorted({m.path for m in psutil.Process(process.pid).memory_maps()
                     if Path(m.path).name.lower() in ('vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll')})
        if any(Path(p).resolve().parent!=folder for p in dlls): raise RuntimeError(f'Non-local CRT DLL: {dlls}')
        result.update(passed=True,browser_processes=len(browser_paths),browser_paths=browser_paths,crt_dlls=dlls)
    except Exception as error:
        result['error']=str(error)
    finally:
        if process.poll() is None:
            subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],capture_output=True)
        process.wait(timeout=30)
        report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__':main()
