"""README comparison: official llama.cpp/Q4_K_M versus production vLLM via JSON API.

Both sides receive stream=false. Each configuration uses fresh server processes;
trial order alternates. Reports retain first workload pass and warmed passes.
"""
import argparse
import hashlib
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
from prepare_official_baseline import MODEL, MODEL_SHA, digest
from vllm_runtime import ROOT, environment, server_command, stop_process_tree


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Continue matching matrix, preserving failed attempts')
    parser.add_argument('--cooldown',type=float,default=120,help='Seconds between fresh processes, outside all timings')
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--client',choices=('urllib','aiohttp','aiohttp-winloop'),default='aiohttp')
    parser.add_argument('--parallels',nargs='+',type=int,choices=(32,256),default=[32,256])
    parser.add_argument('--backends',nargs='+',choices=('llama','vllm'),default=['llama','vllm'])
    parser.add_argument('--llama-kv',choices=('f16','q8_0'),default='f16')
    parser.add_argument('--port',type=int,default=18086)
    parser.add_argument('--runtime-root',type=Path,default=ROOT/'desktop/src-tauri/resources/payload')
    args=parser.parse_args()
    if args.cooldown<0:parser.error('cooldown must be nonnegative')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=args.resume)
    baseline=ROOT/'.local/official-api-baseline';binary=baseline/'bin'
    provenance=json.loads((baseline/'provenance.json').read_text(encoding='utf-8'))
    for name,expected in provenance['unmodified_extracted_files'].items():
        if digest(binary/name)!=expected:raise ValueError('Modified upstream artifact: '+name)
    if digest(baseline/MODEL)!=MODEL_SHA:raise ValueError('Modified official model')
    runtime=args.runtime_root.resolve()
    clean={k:v for k,v in os.environ.items() if not k.upper().startswith(('LLAMA_','GGML_','CUDA','VLLM','PYTHON','BENCHMARK_API_KEY'))}
    clean['PATH']=os.pathsep.join([str(binary),str(ROOT/'runtime/vllm'),str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32')])
    clean['CUDA_VISIBLE_DEVICES']='0'
    clean['CUDA_CACHE_PATH']=str(output/'llama-cuda-cache')
    version=subprocess.run([str(binary/'llama-server.exe'),'--version'],env=clean,capture_output=True,check=True)
    version=(version.stdout+version.stderr).decode('utf-8',errors='replace')
    plan={p:resolve_config(ROOT,profile='fast',parallel=p,context=2048) for p in args.parallels}
    result={'schema_version':1,'protocol':'non_streaming_json','client':args.client,'upstream':provenance,'upstream_version':version,
        'vllm_runtime':json.loads((runtime/'runtime/vllm/hymt-runtime.json').read_text()),
        'vllm_model':json.loads((ROOT/'models/manifest.json').read_text()),
        'gpu':plan[args.parallels[0]]['gpu'],'trials':args.trials,'warm_repeats':args.repeats,
        'context_per_request':2048,'batch_tokens':2048,'llama_kv':args.llama_kv,
        'kv_token_capacity':{p:plan[p]['budget']['kv_token_capacity'] for p in args.parallels},
        'vllm_kv':'int8_per_token_head','runs':[],
        'notes':['Untouched official stable v0.4.1 release binaries (nightly b10964) and untouched Tencent Q4_K_M.',
                 'Complete translation deployment comparison; weight format, activation precision, KV format and allocation differ.',
                 'Both use loopback POST /v1/chat/completions with stream=false; identical prompts and sampling settings.',
                 '512 requests, bounded client task concurrency; server startup and output file writing excluded.',
                 'Both have prefix/prompt caching disabled. First complete workload pass reported separately.',
                 'Both use shared KV pools with equal token capacity and a 2048-token per-request context limit.',
                 'Fresh server process per trial/configuration; backend and concurrency order alternate.',
                 'Unchanged 64 bilingual cases repeated eight times; not a production traffic or length distribution.']}
    torch_root=runtime/'runtime/vllm/Lib/site-packages/torch'
    marker=torch_root/'hymt-build.json'
    result['runtime_identity']={'torch_cuda_sha256':digest(torch_root/'lib/torch_cuda.dll'),
                                'custom_build_marker_present':marker.is_file()}
    if marker.is_file():
        build=json.loads(marker.read_text(encoding='utf-8'))
        result['runtime_identity'].update(profile=build['profile'],inference_validated=build['inference_validated'])
    if args.resume:
        previous=json.loads((output/'matrix.json').read_text(encoding='utf-8'))
        if previous.get('client','urllib')!=args.client:raise ValueError('Resume mismatch: client')
        if 'runtime_identity' in previous and previous['runtime_identity']!=result['runtime_identity']:
            raise ValueError('Resume mismatch: runtime binary identity')
        for key in ('upstream','upstream_version','vllm_runtime','vllm_model','trials','warm_repeats',
                    'context_per_request','batch_tokens','llama_kv','kv_token_capacity','vllm_kv'):
            # JSON object keys become strings on disk.
            if previous[key]!=json.loads(json.dumps(result[key])):raise ValueError('Resume mismatch: '+key)
        result=previous
    result.setdefault('cooldown_history',[]).append({'after_completed_runs':len(result['runs']),'seconds':args.cooldown})
    save=lambda: (output/'matrix.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    save()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for trial in range(1,args.trials+1):
        backends=args.backends if trial%2 else list(reversed(args.backends))
        parallels=args.parallels if trial%2 else list(reversed(args.parallels))
        for parallel in parallels:
            for backend in backends:
                name=f'{backend}-p{parallel}-t{trial}'
                if any(r['backend']==backend and r['parallel']==parallel and r['trial']==trial for r in result['runs']):continue
                if (output/name).exists():
                    failed_report=json.loads((output/name/'report.json').read_text())
                    failures=result.setdefault('failed_attempts',[])
                    if not any(r['name']==name for r in failures):
                        errors=[]
                        for phase in failed_report['runs']:
                            rows=[json.loads(line) for line in (output/name/(phase['phase']+'.jsonl')).read_text(encoding='utf-8').splitlines()]
                            errors.extend({'id':r['id'],'error':r['error']} for r in rows if not r['ok'])
                        failures.append({'name':name,'backend':backend,'parallel':parallel,'trial':trial,'report':failed_report,'errors':errors})
                        save()
                    base=name;attempt=1
                    while (output/name).exists():
                        name=f'{base}-retry{attempt}';attempt+=1
                if result['runs'] and args.cooldown:
                    print(f'Connection cooldown: {args.cooldown:g}s before {name}',flush=True)
                    time.sleep(args.cooldown)
                print('Starting '+name,flush=True)
                with socket.socket() as probe:probe.bind(('127.0.0.1',args.port))
                if backend=='llama':
                    command=[str(binary/'llama-server.exe'),'-m',str(baseline/MODEL),'--alias','hy-mt2',
                        '--host','127.0.0.1','--port',str(args.port),'-ngl','all','-fa','on',
                        '-c',str(plan[parallel]['budget']['kv_token_capacity']),'-np',str(parallel),'-b','2048','-ub','2048',
                        '--kv-unified','--kv-unified-per-slot','2048','--cache-ram','0',
                        '-t','8','-tb','8','--jinja','-ctk',args.llama_kv,'-ctv',args.llama_kv]
                    env=clean
                else:
                    command=server_command(plan[parallel],root=runtime,port=args.port)
                    env=environment(runtime,cache=output/'vllm-cache',gpu=plan[parallel]['gpu']['uuid'])
                record={'name':name,'backend':backend,'parallel':parallel,'trial':trial,'command':command}
                started=time.monotonic()
                with telemetry(output/(name+'.gpu.jsonl')),(output/(name+'.stdout.log')).open('wb') as stdout,(output/(name+'.stderr.log')).open('wb') as stderr:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        deadline=time.monotonic()+600
                        while True:
                            if process.poll() is not None:raise RuntimeError(f'{name} server exited: {process.returncode}')
                            try:
                                with opener.open(f'http://127.0.0.1:{args.port}/health',timeout=2) as response:
                                    if response.status==200:break
                            except OSError:pass
                            if time.monotonic()>deadline:raise TimeoutError(name+' startup timeout')
                            time.sleep(.5)
                        record['initialization_s']=time.monotonic()-started
                        with (output/(name+'.client.log')).open('wb') as log:
                            subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_backend.py'),'--backend',backend,
                                '--url',f'http://127.0.0.1:{args.port}','--concurrency',str(parallel),'--max-tokens','512',
                                '--client',args.client,'--repeats',str(args.repeats),'--output',str(output/name)],
                                env={k:v for k,v in os.environ.items() if k!='BENCHMARK_API_KEY'},stdout=log,stderr=log,check=True,timeout=1800)
                        record['report']=json.loads((output/name/'report.json').read_text())
                    finally:stop_process_tree(process)
                result['runs'].append(record);save()
                print('Passed '+name+': '+json.dumps(record['report']['warm_median']),flush=True)
    print('Complete '+str(output/'matrix.json'),flush=True)


if __name__=='__main__':main()
