"""Translate JSONL directly with the native Windows vLLM engine (no HTTP)."""
import argparse,json,subprocess,sys
from pathlib import Path
from gpu_config import resolve_config
from vllm_runtime import environment,python_executable,llm_config
ROOT=Path(__file__).resolve().parents[1]
def check_distinct_paths(paths):
    items = list(paths.items())
    for index, (name, path) in enumerate(items):
        for other_name, other in items[:index]:
            if path == other or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError(f'{name} and {other_name} must be different files: {path}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path,nargs='?')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--profile',choices=('auto','fast','quality','compat'),default='auto')
    parser.add_argument('--parallel',type=int,default=32)
    parser.add_argument('--context',type=int,default=2048)
    parser.add_argument('--batch-tokens',type=int,default=2048)
    parser.add_argument('--kv-cache-dtype',choices=('int8_per_token_head','bfloat16','fp8_per_token_head'),default='int8_per_token_head')
    parser.add_argument('--kv-gib',type=float)
    parser.add_argument('--memory-percent',type=int,default=75)
    parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--greedy',action='store_true')
    parser.add_argument('--gpu')
    args=parser.parse_args()
    source=args.input.resolve()
    destination=(args.output or source.with_name(source.stem+'.translated.jsonl')).resolve()
    summary=destination.with_suffix('.summary.json')
    config=destination.with_suffix('.config.json')
    log=destination.with_suffix('.log')
    check_distinct_paths({'input':source,'output':destination,'summary':summary,'config':config,'log':log})
    for path in (destination,summary,config,log):
        if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
    cases=[json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not cases or args.max_tokens<1 or args.max_tokens>=args.context: parser.error('Invalid requests or token budget')
    if len({str(x['id']) for x in cases})!=len(cases): parser.error('Duplicate request IDs')
    plan=resolve_config(ROOT,mode='batch',profile=args.profile,model_override=args.model,parallel=args.parallel,
        context=args.context,ubatch=args.batch_tokens,gpu=args.gpu,cache_type_k=args.kv_cache_dtype,
        kv_gib=args.kv_gib,memory_percent=args.memory_percent)
    for path in (destination,summary,config,log):
        if path.is_relative_to(Path(plan['model'])): raise ValueError('Output cannot overwrite model files')
    destination.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps({'plan':plan,'llm':llm_config(plan)},indent=2),encoding='utf-8')
    command=[str(python_executable(ROOT)),str(ROOT/'scripts/vllm_batch_worker.py'),str(config),str(source),str(destination),str(summary),str(args.max_tokens),str(args.seed)]
    if args.greedy: command.append('--greedy')
    with log.open('wb') as stream:
        result=subprocess.run(command,env=environment(ROOT,gpu=plan['gpu']['uuid']),stdout=stream,stderr=stream,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode: print(log.read_text(encoding='utf-8',errors='replace')[-4000:],file=sys.stderr)
    else: print(summary.read_text(encoding='utf-8'))
    return result.returncode

if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError) as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(1)
