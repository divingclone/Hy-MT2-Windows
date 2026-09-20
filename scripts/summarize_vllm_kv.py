"""Validate and aggregate benchmark_vllm_kv.py output without raw prompts/paths."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as st


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {'median':st.median(values), 'min':min(values), 'max':max(values)}


def summarize(source):
    groups = {}
    failures = []
    expected_prompts = None
    for process in sorted(source.glob('*.process.json')):
        name=process.name.removesuffix('.process.json')
        match=re.fullmatch(r'(.+)-p(32|256)-t(\d+)',name)
        if not match:
            continue
        variant, parallel, trial=match[1],int(match[2]),int(match[3])
        proc=read(process)
        if proc['exit_code']:
            failures.append({'name':name,'exit_code':proc['exit_code'],
                'log_sha256':digest(source/(name+'.log'))})
            continue
        folder=source/name
        native=variant=='llama-q8'
        prompts=read(folder/('summary.prompts.json' if native else 'prompt_ids.json'))
        if expected_prompts is None:
            expected_prompts=prompts
        assert prompts==expected_prompts, f'{name}: prompt token IDs differ'
        if native:
            phases=[read(folder/'summary.cold.json')]+[read(p) for p in sorted(folder.glob('summary.warm-*.json'))]
            initialization=phases[0]['initialization_s']
            config={k:phases[0][k] for k in ('parallel','context_per_sequence','max_tokens','cache_type_k','cache_type_v','batch','ubatch')}
        else:
            report=read(folder/'report.json')
            assert report['metric_profile']=='pretokenized_engine'
            phases=report['runs']
            initialization=report['initialization_s']
            config={k:v for k,v in report['config'].items() if k!='model'}
        gpu=lines(source/(name+'.gpu.jsonl'))
        log=(source/(name+'.log')).read_text(encoding='utf-8', errors='replace')
        capacities=re.findall(r'GPU KV cache size: ([\d,]+) tokens',log)
        backend_lines=[s for s in log.splitlines() if 'attention backend out of' in s or 'Using AttentionBackendEnum.' in s]
        runs=[]
        for phase in phases:
            label=phase['phase']
            path=folder/(f'output.{label}.jsonl' if native else f'{label}.jsonl')
            outputs=lines(path)
            assert len(outputs)==len(prompts)
            assert [r['id'] for r in outputs]==[p['id'] for p in prompts]
            assert all(r['ok'] and not r['truncated'] and r['translation'].strip() and r['finish_reason']=='stop' for r in outputs)
            assert sum(r['prompt_tokens'] for r in outputs)==sum(len(p['token_ids']) for p in prompts)
            assert sum(r['completion_tokens'] for r in outputs)==phase['completion_tokens']
            if not native:
                assert all(r['terminal_eos_present'] for r in outputs)
                assert phase['output_sha256']==digest(path)
            wall=phase['core_wall_s'] if native else phase['wall_s']
            samples=[r for r in gpu if phase['phase_start_unix_s']<=r['received_unix_s']<=phase['phase_end_unix_s']]
            assert samples
            runs.append({'phase':label,'core_wall_s':wall,'completion_tokens':phase['completion_tokens'],
                'completion_tokens_per_second':phase['completion_tokens']/wall,'requests':len(outputs),
                'errors':0,'truncated':0,'empty_outputs':0,'output_sha256':digest(path),
                'phase_gpu':{k:stats([float(r[k]) for r in samples]) for k in
                    ('clocks.sm','power.draw','utilization.gpu','memory.used')},
                'source_report_sha256':digest(folder/(f'summary.{label}.json' if native else 'report.json'))})
        baseline=float(gpu[0]['memory.used'])
        peak=max(float(r['memory.used']) for r in gpu)
        entry={'trial':trial,'config':config,'initialization_s':initialization,
            'cache_capacity_tokens':int(capacities[-1].replace(',','')) if capacities else None,
            'attention_backend':backend_lines[-1].split('] ')[-1] if backend_lines else 'native',
            'preemption_log_lines':sum('preempt' in s.lower() and ('WARNING' in s or 'Preempt' in s) for s in log.splitlines()),
            'memory_baseline_mib':baseline,'memory_peak_mib':peak,'memory_peak_delta_gib':(peak-baseline)/1024,
            'telemetry_sha256':digest(source/(name+'.gpu.jsonl')),
            'warm_median_tps':st.median(r['completion_tokens_per_second'] for r in runs[1:]),
            'warm_median_wall_s':st.median(r['core_wall_s'] for r in runs[1:]),'runs':runs}
        groups.setdefault((variant,parallel),[]).append(entry)
    summaries=[]
    for (variant,parallel),trials in groups.items():
        summaries.append({'variant':variant,'parallel':parallel,'trials':trials,
            'warm_tps':stats([t['warm_median_tps'] for t in trials]),
            'warm_wall_s':stats([t['warm_median_wall_s'] for t in trials]),
            'memory_peak_delta_gib':stats([t['memory_peak_delta_gib'] for t in trials]),
            'initialization_s':stats([t['initialization_s'] for t in trials])})
    for group in summaries:
        for baseline in ('bf16','llama-q8'):
            reference=next((g for g in summaries if g['variant']==baseline and g['parallel']==group['parallel']),None)
            if reference:
                group[f'throughput_ratio_to_{baseline}']=group['warm_tps']['median']/reference['warm_tps']['median']
    protocol=read(source/'protocol.json')
    if (source/'input.jsonl').exists():
        protocol['input_sha256']=digest(source/'input.jsonl')
    return {'schema_version':1,'protocol':protocol,
        'aggregation':'Median of process warm medians. First pass/init separate. Failed processes excluded and listed.',
        'prompt_token_ids_equal':True if expected_prompts else None,
        'prompt_tokens':sum(len(p['token_ids']) for p in expected_prompts) if expected_prompts else None,
        'groups':summaries,'failed_processes':failures,
        'memory_note':'Device-wide 100 ms sampled peak minus first prelaunch sample, including init/graphs; not isolated process memory.',
        'gpu_units':{'clocks.sm':'MHz','power.draw':'W','utilization.gpu':'percent','memory.used':'MiB'},
        'preemption_note':'Log inspection only; absence of warnings is not proof of zero preemptions.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    for group in result['groups']:
        print(group['variant'],group['parallel'],round(group['warm_tps']['median'],2),
              round(group['memory_peak_delta_gib']['median'],2),'GiB')


if __name__=='__main__':
    main()
