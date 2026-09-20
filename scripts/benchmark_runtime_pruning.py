"""Compare two release runtimes at 32/256 concurrency, including exact token IDs."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from gpu_config import resolve_config
from vllm_runtime import environment,llm_config,python_executable,stop_process_tree

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-root',required=True,type=Path)
parser.add_argument('--candidate-root',required=True,type=Path)
parser.add_argument('--model-root',default=ROOT,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--baseline-cudnn-config',choices=('FULL','GRAPH_JIT_ONLY'),default='GRAPH_JIT_ONLY',
                    help='Use FULL only when comparing against the original unpruned cuDNN runtime')
parser.add_argument('--repeats',type=int,default=3)
parser.add_argument('--candidate-first',action='store_true',help='Reverse run order for an independent confirmation')
parser.add_argument('--profiles',nargs='+',choices=('fast','compat'),default=['fast','compat'])
parser.add_argument('--parallels',nargs='+',type=int,choices=(32,256),default=[32,256])
args=parser.parse_args()
if args.repeats<1: parser.error('--repeats must be positive')
OUT=args.output.resolve()
OUT.mkdir(exist_ok=False)
roots={'baseline':args.baseline_root.resolve(),'candidate':args.candidate_root.resolve()}
roots={k:v/'payload' if (v/'payload').is_dir() else v for k,v in roots.items()}
summary={'configs':{},'runs':{},'comparisons':{},'valid':False,
         'baseline_cudnn_config':args.baseline_cudnn_config,'warm_repeats':args.repeats}
summary['order']=['candidate','baseline'] if args.candidate_first else ['baseline','candidate']
for profile in args.profiles:
 for parallel in args.parallels:
  key=f'{profile}-p{parallel}'
  plan=resolve_config(args.model_root.resolve(),profile=profile,parallel=parallel,context=2048,kv_gib=1.546875 if parallel==32 else 3.09375)
  config=OUT/(key+'.json'); config.write_text(json.dumps(llm_config(plan)),encoding='utf-8')
  summary['configs'][key]=llm_config(plan)
  for variant in summary['order']:
   name=key+'-'+variant;root=roots[variant];env=environment(root,cache=OUT/('cache-'+variant))
   env['CUDNN_LIB_CONFIG']=args.baseline_cudnn_config if variant=='baseline' else 'GRAPH_JIT_ONLY'
   command=[str(python_executable(root)),'-B','-s','-X','utf8',str(ROOT/'scripts/benchmark_vllm_offline.py'),
            '--config',str(config),'--output',str(OUT/name),'--pretokenized','--save-token-ids','--repeats',str(args.repeats),'--max-tokens','512']
   print('Starting '+name,flush=True)
   with (OUT/(name+'.log')).open('xb') as log:
    process=subprocess.Popen(command,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    try: code=process.wait(timeout=900)
    finally:
     if process.poll() is None:stop_process_tree(process)
   if code:raise RuntimeError(name+' failed; see its log')
   report=json.loads((OUT/name/'report.json').read_text())
   assert all(r['valid'] for r in report['runs'])
   summary['runs'][name]={'warm_median':report['warm_median'],'runs':report['runs'],
                         'initialization_s':report['initialization_s']}
   print('Passed '+name+': '+str(report['warm_median']),flush=True)
  equality=[]
  for phase in ('cold',*(f'warm-{i}' for i in range(1,args.repeats+1))):
   left=[json.loads(s) for s in (OUT/(key+'-baseline')/(phase+'.jsonl')).read_text(encoding='utf-8').splitlines()]
   right=[json.loads(s) for s in (OUT/(key+'-candidate')/(phase+'.jsonl')).read_text(encoding='utf-8').splitlines()]
   keys=('id','translation','token_ids','completion_tokens','prompt_tokens','finish_reason')
   matches=sum(all(a[k]==b[k] for k in keys) for a,b in zip(left,right))
   equality.append({'phase':phase,'matched':matches,'requests':len(left)})
   assert len(left)==len(right)==512 and matches==512,(key,phase,matches)
  summary['comparisons'][key]=equality
  (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
summary['valid']=True
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
paired=len(summary['comparisons'])*512*(args.repeats+1)
print(f'All {2*paired:,} requests valid; {paired:,} paired outputs identical.',flush=True)
