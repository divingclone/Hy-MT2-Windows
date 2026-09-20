"""Cold non-greedy GPU smoke with host build tools excluded and audited.

This is a dependency/startup test, not a throughput benchmark. Each invocation
requires a fresh output directory and uses production environment construction.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_config import resolve_config
from vllm_runtime import environment, llm_config, python_executable, stop_process_tree

AUDIT = '''import atexit,json,os,subprocess
from pathlib import Path
_original=subprocess.Popen
class AuditedPopen(_original):
    def __init__(self,args,*pos,**kw):
        command=args if isinstance(args,str) else " ".join(str(x) for x in args)
        low=command.lower()
        forbidden=any(x in low for x in ("cl.exe","vswhere.exe","microsoft visual studio","windows kits","nvidia gpu computing toolkit"))
        # FlashInfer may probe the absent bundled nvcc for its version, then
        # fall back to torch.version.cuda. It must never find a real compiler.
        if not isinstance(args,str) and "nvcc" in Path(str(args[0])).name.lower():
            forbidden=forbidden or Path(str(args[0])).exists()
        with open(os.environ["HYMT_COMPILER_AUDIT"],"a",encoding="utf-8") as f:
            f.write(json.dumps({"pid":os.getpid(),"command":command,"forbidden":forbidden})+"\\n")
        if forbidden: raise RuntimeError("Host build tools forbidden: "+command)
        super().__init__(args,*pos,**kw)
subprocess.Popen=AuditedPopen
@atexit.register
def loaded_libraries():
    try:
        import psutil
        paths=sorted({m.path for m in psutil.Process().memory_maps() if m.path.lower().endswith((".dll",".pyd"))})
        Path(os.environ["HYMT_COMPILER_AUDIT"]+"."+str(os.getpid())+".dlls.json").write_text(json.dumps(paths,indent=2),encoding="utf-8")
    except Exception: pass
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--profile',choices=('fast','quality','compat'),default='fast')
    parser.add_argument('--model-root',type=Path,help='Use verified external weights while testing a model-free staged runtime')
    parser.add_argument('--parallel',type=int,default=32)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    if (root/'payload').is_dir(): root/='payload'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=resolve_config(args.model_root.resolve() if args.model_root else root,profile=args.profile,parallel=args.parallel)
    env=environment(root,cache=out/'cache',gpu=plan['gpu']['uuid'])
    audit=out/'compiler-audit.jsonl'
    env.update(PYTHONPATH=str(out),HYMT_COMPILER_AUDIT=str(audit))
    (out/'sitecustomize.py').write_text(AUDIT,encoding='utf-8')
    config=out/'config.json'
    config.write_text(json.dumps({'plan':plan,'llm':llm_config(plan)},indent=2),encoding='utf-8')
    command=[str(python_executable(root)),'-u','-s','-X','utf8',str(root/'scripts/vllm_batch_worker.py'),
             str(config),str(root/'examples/input.jsonl'),str(out/'outputs.jsonl'),str(out/'summary.json'),'128','42']
    (out/'environment.json').write_text(json.dumps({k:env[k] for k in ('PATH','CC','CXX','CUDA_PATH','CUDA_HOME','VLLM_USE_FLASHINFER_SAMPLER','FLASHINFER_DISABLE_JIT')},indent=2),encoding='utf-8')
    with (out/'run.log').open('wb') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=log)
        try: code=process.wait(timeout=600)
        finally:
            if process.poll() is None: stop_process_tree(process)
    records=[json.loads(x) for x in audit.read_text(encoding='utf-8').splitlines()] if audit.exists() else []
    passed=code==0 and bool(records) and not any(r['forbidden'] for r in records)
    report={'passed':passed,'exit_code':code,'profile':args.profile,'parallel':args.parallel,
            'host_build_tools_excluded':True,'cold_cache':True,'flashinfer_jit_disabled':True,
            'gpu':plan['gpu'],'subprocess_count':len(records),'throughput_benchmark':False}
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not passed: raise SystemExit(1)


if __name__=='__main__':main()
