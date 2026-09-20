"""Validate transport/tuning screens and export aggregate, path-free evidence."""
import hashlib
import json
from pathlib import Path
import statistics
from benchmark_inputs import ROOT


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    screens=[]
    for name in ('api-opt-screen-p256','api-opt-front2','api-opt-batch4096',
                 'api-opt-batch1024','api-opt-batch1024-recheck','api-opt-screen-p32'):
        directory=ROOT/'results'/name
        matrix=json.loads((directory/'matrix.json').read_text(encoding='utf-8'))
        groups={};passes=[]
        for run in matrix['runs']:
            report=run['report'];assert report['protocol']=='non_streaming_json'
            warm=[]
            for phase in report['runs']:
                path=directory/run['name']/(phase['phase']+'.jsonl')
                assert digest(path)==phase['output_sha256']
                rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
                assert len(rows)==len({r['id'] for r in rows})==512
                assert all(r['ok'] and not r['truncated'] and r['finish_reason']=='stop' for r in rows)
                assert sum(r['completion_tokens'] for r in rows)==phase['metrics']['completion_tokens']
                assert phase['metrics']['valid']
                item={'client':run['client'],'trial':run['trial'],**phase}
                passes.append(item)
                if phase['phase']!='cold':warm.append(phase['metrics'])
            med={k:statistics.median(r[k] for r in warm) for k in ('wall_s','completion_tokens_per_second','requests_per_second')}
            med['latency_p95_s']=statistics.median(r['latency_s']['p95'] for r in warm)
            groups.setdefault(run['client'],[]).append({'trial':run['trial'],**med})
        summary=[]
        for client,trials in groups.items():
            summary.append({'client':client,'trials':trials,
                'median':{k:statistics.median(r[k] for r in trials) for k in trials[0] if k!='trial'}})
        entry={'name':name,'concurrency':matrix['plan']['parallel'],'batch_tokens':matrix['plan']['batch'],
            'summary':summary,'passes':passes,'matrix_sha256':digest(directory/'matrix.json')}
        if name=='api-opt-front2':
            assert not matrix['runs']
            error=(directory/'server.stderr.log').read_text(encoding='utf-8',errors='replace')
            assert 'SO_REUSEPORT' in error
            entry['failure']='Windows frontend scale-out startup requires missing socket.SO_REUSEPORT; no valid performance result.'
            entry['failure_log_sha256']=digest(directory/'server.stderr.log')
        if name=='api-opt-batch1024':
            entry['excluded_from_selection']='CPU unit tests overlapped this exploratory run; use the isolated recheck.'
        if 'core' in matrix:
            core=matrix['core'];entry['matched_core']=core['warm_median']
            entry['core_passes']=core['runs']
            for phase in core['runs']:
                assert digest(directory/'core'/(phase['phase']+'.jsonl'))==phase['output_sha256']
                assert phase['valid']
        screens.append(entry)
    smoke_path=ROOT/'results/api-opt-production-output.jsonl'
    inputs=[json.loads(x) for x in (ROOT/'results/api-opt-production-input.jsonl').read_text(encoding='utf-8').splitlines()]
    smoke=[json.loads(x) for x in smoke_path.read_text(encoding='utf-8').splitlines()]
    assert len(smoke)==512 and [r['id'] for r in smoke]==[r['id'] for r in inputs]
    assert all(r['ok'] and not r['truncated'] and r['finish_reason']=='stop' and r['seed']==42+i for i,r in enumerate(smoke))
    result={'schema_version':1,'hardware':'RTX 5090 / AMD Ryzen 7 9800X3D / Windows native',
        'protocol':'non_streaming_json','context':2048,'requests_per_pass':512,
        'sampling':'Unchanged: temperature .7, top_p .6, top_k 20, repetition_penalty 1.05, max_tokens 512, seed 42 + index',
        'selected':'aiohttp + asyncio pooled client; existing single vLLM API frontend; batch_tokens 2048',
        'notes':['Screens compare multiple clients in one server process, with order reversed in the second round.',
                 'Each round has a first workload pass followed by 3 warm passes. Selection is exploratory, not a confidence interval.',
                 'Final official baseline comparison uses separate independent processes and is published separately.',
                 'Core rerun uses the same 2048-context runtime/configuration; historical 1024-context TPS is not the denominator.',
                 'No changes to model weights, KV precision, sampling or maximum output length.'],
        'screens':screens,'packaged_client_smoke':{'runtime':'release staging runtime and scripts',
            'concurrency':32,'requests':512,'errors':0,'truncated':0,'ids_and_order_and_seeds_verified':True,
            'output_sha256':digest(smoke_path),'note':'Functional validation after service restoration, not part of throughput comparison.'}}
    out=ROOT/'benchmarks/api-optimization.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'name':x['name'],'summary':x['summary']} for x in screens],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
