"""32/256 core throughput and INT4/BF16-teacher fidelity in the portable runtime.

Run with no other GPU workloads. Output must be new. All inference uses the
production compiler-free environment; initialization is excluded from TPS.
"""
import argparse
import json
from pathlib import Path
import subprocess

from gpu_config import resolve_config
from vllm_runtime import environment, llm_config, python_executable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=False)
    env=environment(root,cache=dest/'cache')
    python=str(python_executable(root))
    def run(name,command):
        print('Starting '+name,flush=True)
        with (dest/(name+'.log')).open('wb') as log:
            subprocess.run(command,env=env,stdout=log,stderr=log,check=True,timeout=900)
        print('Passed '+name,flush=True)
    run('int4-cold',[python,str(root/'scripts/validate_portable_toolchain.py'),'--profile','compat','--output',str(dest/'int4-cold')])
    for profile in ('fast','compat'):
        for parallel in (32,256):
            name=f'{profile}-p{parallel}'
            cfg=llm_config(resolve_config(root,profile=profile,parallel=parallel,context=1024))
            path=dest/(name+'.json');path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            run(name,[python,str(root/'scripts/benchmark_vllm_offline.py'),'--config',str(path),
                '--output',str(dest/name),'--pretokenized','--repeats','3','--max-tokens','512'])
    name='int4-teacher-p32'
    cfg=llm_config(resolve_config(root,profile='compat',parallel=32,context=4096))
    path=dest/(name+'.json');path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    run(name,[python,str(root/'scripts/benchmark_vllm_offline.py'),'--config',str(path),
        '--input',str(root/'results/opus-v2/evaluation.jsonl'),'--output',str(dest/name),
        '--pretokenized','--greedy','--repeats','1','--max-tokens','2048','--save-token-ids',
        '--teacher-output',str(root/'results/teacher-fidelity/teacher-p32/warm-1.jsonl')])
    summary={name:json.loads((dest/name/'report.json').read_text(encoding='utf-8'))['warm_median']
             for name in ('fast-p32','fast-p256','compat-p32','compat-p256')}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
