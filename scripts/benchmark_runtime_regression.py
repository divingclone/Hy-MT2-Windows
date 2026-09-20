"""Repeat the same 32/256 core benchmark using a candidate release runtime."""
import argparse
import json
from pathlib import Path
import subprocess

from gpu_config import resolve_config
from vllm_runtime import ROOT, environment, llm_config, python_executable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profiles',nargs='+',choices=('fast','compat'),default=['fast','compat'])
    parser.add_argument('--parallels',nargs='+',type=int,choices=(32,256),default=[32,256])
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    root=args.root.resolve()
    if (root/'payload').is_dir():root/='payload'
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    summary={}
    for profile in args.profiles:
        for parallel in args.parallels:
            name=f'{profile}-p{parallel}'
            config=output/(name+'.json')
            config.write_text(json.dumps(llm_config(resolve_config(ROOT,profile=profile,parallel=parallel,context=1024)),indent=2),encoding='utf-8')
            env=environment(root,cache=output/'cache')
            print(f'Starting {name}',flush=True)
            with (output/(name+'.log')).open('wb') as log:
                subprocess.run([str(python_executable(root)),'-s','-X','utf8',str(ROOT/'scripts/benchmark_vllm_offline.py'),
                    '--config',str(config),'--output',str(output/name),'--pretokenized','--repeats',str(args.repeats),'--max-tokens','512'],
                    env=env,stdout=log,stderr=log,check=True,timeout=900)
            report=json.loads((output/name/'report.json').read_text(encoding='utf-8'))
            summary[name]=report['warm_median']
            print(f'Passed {name}: {summary[name]}',flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
