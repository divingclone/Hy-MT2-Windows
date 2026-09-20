"""Controlled native Windows API transport/front-end screen; never overwrite runs."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
from benchmark_common import telemetry
from gpu_config import resolve_config
from vllm_runtime import ROOT,environment,server_command,stop_process_tree


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parallel',type=int,choices=(32,256),default=256)
    p.add_argument('--api-servers',type=int,default=1)
    p.add_argument('--batch',type=int,default=2048)
    p.add_argument('--clients',nargs='+',default=['urllib','aiohttp','aiohttp-winloop','aiohttp-winloop-close'])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--core',action='store_true')
    p.add_argument('--port',type=int,default=18086)
    p.add_argument('--server-args',nargs=argparse.REMAINDER,default=[])
    a=p.parse_args();d=a.output.resolve();d.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',a.port))
    plan=resolve_config(ROOT,profile='fast',parallel=a.parallel,context=2048,ubatch=a.batch)
    env=environment(ROOT,cache=ROOT/'cache/api-optimization',gpu=plan['gpu']['uuid'])
    cmd=server_command(plan,port=a.port)+['--api-server-count',str(a.api_servers)]+a.server_args
    report={'plan':plan,'command':cmd,'runs':[],'errors':[]}
    save=lambda:(d/'matrix.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started=time.perf_counter()
    with telemetry(d/'gpu.jsonl'),(d/'server.stdout.log').open('wb') as out,(d/'server.stderr.log').open('wb') as err:
        proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+600
            while True:
                if proc.poll() is not None:raise RuntimeError('Server exited '+str(proc.returncode))
                try:
                    with op.open(f'http://127.0.0.1:{a.port}/health',timeout=2) as response:
                        if response.status==200:break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('Server startup')
                time.sleep(.5)
            report['initialization_s']=time.perf_counter()-started;save()
            for trial in range(1,a.rounds+1):
                clients=a.clients if trial%2 else list(reversed(a.clients))
                for client in clients:
                    name=f'{client}-t{trial}';close=client.endswith('-close');kind=client.removesuffix('-close')
                    command=[sys.executable,'-X','utf8',str(ROOT/'scripts/benchmark_backend.py'),
                        '--backend','vllm','--url',f'http://127.0.0.1:{a.port}','--concurrency',str(a.parallel),
                        '--client',kind,'--repeats',str(a.repeats),'--output',str(d/name)]
                    if close:command+=['--no-keepalive']
                    print('Starting '+name,flush=True)
                    with (d/(name+'.log')).open('wb') as log:
                        done=subprocess.run(command,stdout=log,stderr=log,timeout=1200,
                            env={k:v for k,v in os.environ.items() if k!='BENCHMARK_API_KEY'})
                    r=json.loads((d/name/'report.json').read_text(encoding='utf-8')) if (d/name/'report.json').exists() else None
                    record={'name':name,'client':client,'trial':trial,'exit_code':done.returncode,'report':r}
                    if done.returncode:report['errors'].append(record)
                    else:report['runs'].append(record)
                    save();print(json.dumps({'name':name,'result':r.get('warm_median') if r else None,'exit_code':done.returncode}),flush=True)
        finally:stop_process_tree(proc)
    if a.core:
        from vllm_runtime import llm_config
        cfg=d/'core.config.json';cfg.write_text(json.dumps(llm_config(plan)),encoding='utf-8')
        with (d/'core.log').open('wb') as log:
            core_command=[sys.executable,'-X','utf8',str(ROOT/'scripts/benchmark_vllm_offline.py'),
                '--config',str(cfg),'--pretokenized','--repeats',str(a.repeats),'--output',str(d/'core')]
            child=subprocess.Popen(core_command,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                code=child.wait(timeout=1200)
                if code:raise subprocess.CalledProcessError(code,core_command)
            finally:stop_process_tree(child)
        report['core']=json.loads((d/'core/report.json').read_text());save()
        print('Core: '+json.dumps(report['core']['warm_median']),flush=True)
    print('Complete '+str(d),flush=True)


if __name__=='__main__':main()
