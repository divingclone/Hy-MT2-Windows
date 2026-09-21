"""Validate every non-streaming result, then emit a path-free public summary."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from datetime import datetime

from benchmark_inputs import ROOT


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/official-nonstream-api.json')
    args=parser.parse_args()
    source=args.input.resolve();matrix=json.loads((source/'matrix.json').read_text(encoding='utf-8'))
    assert matrix['protocol']=='non_streaming_json'
    assert matrix['trials']==3 and matrix['warm_repeats']==3
    assert len(matrix['runs'])==12
    checkpoint=ROOT/'models'/matrix['vllm_model']['checkpoint_dir']
    for name,expected in matrix['vllm_model']['checkpoint_files'].items():
        path=checkpoint/name
        assert path.stat().st_size==expected['size_bytes'] and digest(path)==expected['sha256']
    groups={};dataset_hashes=set();all_phases=[];memory_measurements=[]
    for run in matrix['runs']:
        report=run['report'];assert report['protocol']=='non_streaming_json'
        assert report['arguments'].get('client','urllib')==matrix.get('client','urllib')
        dataset_hashes.add(report['dataset_sha256'])
        assert report['arguments']['stream'] is False
        assert len(report['runs'])==4
        for phase in report['runs']:
            path=source/run['name']/(phase['phase']+'.jsonl')
            assert digest(path)==phase['output_sha256']
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            metrics=phase['metrics']
            assert len(rows)==512 and len({r['id'] for r in rows})==512
            assert all(r['ok'] and not r['truncated'] and r['finish_reason']=='stop' and r['ttft_s'] is None for r in rows)
            assert sum(r['completion_tokens'] for r in rows)==metrics['completion_tokens']
            assert metrics['valid'] and metrics['observed_peak_in_flight']==run['parallel']
            assert metrics['errors']==metrics['truncated']==0
            assert abs(metrics['completion_tokens']/metrics['wall_s']-metrics['completion_tokens_per_second'])<1e-8
            all_phases.append({'backend':run['backend'],'concurrency':run['parallel'],'trial':run['trial'],
                              'phase':phase['phase'],'metrics':metrics,'output_sha256':phase['output_sha256']})
        warm=[x['metrics'] for x in report['runs'][1:]]
        trace=source/(run['name']+'.gpu.jsonl')
        samples=[float(json.loads(line)['memory.used']) for line in trace.read_text(encoding='utf-8').splitlines()]
        assert samples and all(v>=0 for v in samples)
        baseline=samples[0];peak=max(samples)
        memory_measurements.append({'backend':run['backend'],'concurrency':run['parallel'],'trial':run['trial'],
            'baseline_mib':baseline,'peak_mib':peak,'peak_delta_gib':(peak-baseline)/1024,
            'samples':len(samples),'telemetry_sha256':digest(trace)})
        trial={'trial':run['trial'],'initialization_s':run['initialization_s'],
            'gpu_baseline_mib':baseline,'gpu_peak_mib':peak,'gpu_peak_delta_gib':(peak-baseline)/1024,
            'first_pass_wall_s':report['runs'][0]['metrics']['wall_s'],
            'first_pass_completion_tokens_per_second':report['runs'][0]['metrics']['completion_tokens_per_second'],
            **{k:statistics.median(x[k] for x in warm) for k in ('wall_s','completion_tokens_per_second','requests_per_second')},
            **{f'latency_p{p}_s':statistics.median(x['latency_s'][f'p{p}'] for x in warm) for p in (50,95,99)}}
        groups.setdefault((run['backend'],run['parallel']),[]).append(trial)
    assert len(dataset_hashes)==1
    summary=[]
    for (backend,parallel),trials in sorted(groups.items()):
        assert sorted(x['trial'] for x in trials)==[1,2,3]
        metrics={k:statistics.median(x[k] for x in trials) for k in trials[0] if k!='trial'}
        summary.append({'backend':backend,'concurrency':parallel,'median':metrics,'trials':trials,
            'process_peak_gpu_delta_gib_range':[min(x['gpu_peak_delta_gib'] for x in trials),max(x['gpu_peak_delta_gib'] for x in trials)],
            'process_median_tps_range':[min(x['completion_tokens_per_second'] for x in trials),max(x['completion_tokens_per_second'] for x in trials)]})
    by_key={(x['backend'],x['concurrency']):x['median'] for x in summary}
    comparison=[]
    for parallel in (32,256):
        a,b=by_key['llama',parallel],by_key['vllm',parallel]
        comparison.append({'concurrency':parallel,'throughput_ratio':b['completion_tokens_per_second']/a['completion_tokens_per_second'],
            'request_rate_ratio':b['requests_per_second']/a['requests_per_second'],
            'batch_time_reduction_percent':(1-b['wall_s']/a['wall_s'])*100,
            'p95_response_reduction_percent':(1-b['latency_p95_s']/a['latency_p95_s'])*100})
    failed=[]
    for attempt in matrix.get('failed_attempts',[]):
        phases=[]
        for phase in attempt['report']['runs']:
            path=source/attempt['name']/(phase['phase']+'.jsonl')
            assert digest(path)==phase['output_sha256']
            phases.append({'phase':phase['phase'],'metrics':phase['metrics'],'output_sha256':phase['output_sha256']})
        failed.append({k:attempt[k] for k in ('name','backend','parallel','trial','errors')} | {'phases':phases})
    failed_metrics=[p['metrics'] for a in failed for p in a['phases']]
    result={'schema_version':1,'measurement_date':str(datetime.fromisoformat(matrix['runs'][0]['report']['created_at']).astimezone().date()),
        'measurement_end_date':str(datetime.fromisoformat(matrix['runs'][-1]['report']['created_at']).astimezone().date()),
        'protocol':'non_streaming_json','client':matrix.get('client','urllib'),
        'hardware':{k:v for k,v in matrix['gpu'].items() if k in ('name','compute_capability','driver_version','total_memory_mib')},
        'cpu':'AMD Ryzen 7 9800X3D (8 cores / 16 threads)','os':'Windows x64, native',
        'official_baseline':matrix['upstream'],'official_version_output':matrix['upstream_version'],
        'vllm_runtime':matrix['vllm_runtime'],
        'vllm_checkpoint':{'repo':matrix['vllm_model']['repositories']['fast'],
            'files':matrix['vllm_model']['checkpoint_files']},
        'workload':{'requests_per_pass':512,'unique_source_cases':64,'repetitions_per_case':8,
            'languages':'Chinese to English and English to Chinese','dataset_sha256':next(iter(dataset_hashes)),
            'max_tokens':512,'context_per_request':matrix['context_per_request'],'batch_tokens':matrix['batch_tokens'],
            'kv_token_capacity':matrix['kv_token_capacity'],'llama_kv':matrix['llama_kv'],'vllm_kv':matrix['vllm_kv'],
            'sampling':{'temperature':.7,'top_p':.6,'top_k':20,'min_p':0,'repetition_penalty':1.05,'seed':'42 + request index'},
            'fresh_processes_per_config':3,'warm_passes_per_process':3,'first_passes_reported_separately':True,
            'aggregation':'Median of 3 warm-pass medians from 3 independent server processes, including latency percentile summaries'},
        'summary':summary,'comparison':comparison,'passes':all_phases,'validated_requests':512*len(all_phases),
        'gpu_memory_measurements':memory_measurements,
        'gpu_memory_method':'Device-wide nvidia-smi sampling every 100 ms; whole server-lifecycle peak minus first pre-launch sample, including initialization/graphs/KV/workspace; desktop noise, not process-exclusive allocation. Median over three independent processes.',
        'failed_attempts':failed,'cooldown_history':matrix.get('cooldown_history',[]),
        'total_requests':512*len(all_phases)+sum(p['requests'] for p in failed_metrics),
        'total_errors':sum(p['errors'] for p in failed_metrics),
        'total_truncations':sum(p['truncated'] for p in failed_metrics),'notes':matrix['notes'],
        'detailed_report_sha256':digest(source/'matrix.json')}
    if 'runtime_identity' in matrix:
        result['runtime_identity']=matrix['runtime_identity']
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'comparison':comparison},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
