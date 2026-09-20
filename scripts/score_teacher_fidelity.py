"""Score quantization fidelity to BF16 teacher outputs, never dataset translations.

Requires sacrebleu==2.6.0, numpy and rapidfuzz. Input is a completed directory
from benchmark_teacher_fidelity.py. Strict ID alignment; failures are errors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
from rapidfuzz.distance import Levenshtein


def read(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs: {path}')
    return rows


def identity(path):
    return {'file': path.name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat-check',type=Path,help='Additional W4A4/BF16 KV p32 run directory')
    args=parser.parse_args()
    root=args.input
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    cases=read(Path(protocol['input']))
    ids=[r['id'] for r in cases]
    def aligned(path, valid=False):
        rows=read(path)
        if [r['id'] for r in rows]!=ids:
            raise ValueError(f'IDs/order mismatch: {path}')
        if valid and any(not r['ok'] or r.get('truncated') or not r['translation'].strip() for r in rows):
            raise ValueError(f'Failed/incomplete output: {path}')
        return rows
    teacher=aligned(root/'teacher-p32/warm-1.jsonl',True)
    refs=[r['translation'].strip() for r in teacher]
    tf_teacher=aligned(root/'teacher-p32/teacher-forced.jsonl')
    groups={'overall':list(range(len(cases)))}
    for field in ('target_lang','bucket'):
        for value in sorted({row[field] for row in cases}):
            groups[f'{field}:{value}']=[i for i,row in enumerate(cases) if row[field]==value]
    metric=CHRF(word_order=2)
    report={'schema_version':1,'reference':'Unquantized BF16 model greedy outputs at concurrency 32',
        'dataset_sha256':protocol['input_sha256'],'requests':len(cases),'sacrebleu_version':sacrebleu.__version__,
        'notes':['chrF++ measures wording fidelity, not semantic correctness or a quality-retention percentage.',
                 'Exact match ignores only leading/trailing whitespace.',
                 'Teacher-forced NLL and top1 use raw logits on identical teacher prefixes including EOS; prefill diagnostic, not decode throughput.',
                 'Greedy free generation retains repetition_penalty=1.05; prompt logprobs exclude this penalty.',
                 'Paired 95% percentile intervals use 1000 target-language-stratified bootstrap draws; no multiple-comparison correction.',
                 'llama.cpp also changes weight quantization and embedding precision; its gap is not pure engine error.'],
        'strategies':{},'paired_differences':{}}
    stats_by_name={}
    nll_by_name={}
    prompt_base=json.loads((root/'teacher-p32/prompt_ids.json').read_text())
    for variant,parallel in protocol['variants']:
        name=f'{variant}-p{parallel}'
        folder=root/name
        native=variant.startswith('llama-')
        output=folder/('output.warm-1.jsonl' if native else 'warm-1.jsonl')
        cold=folder/('output.cold.jsonl' if native else 'cold.jsonl')
        if native and not cold.exists():
            cold=folder/'output'
        rows=aligned(output,True)
        earlier=aligned(cold,True)
        prompt_path=folder/('summary.prompts.json' if native else 'prompt_ids.json')
        if json.loads(prompt_path.read_text())!=prompt_base:
            raise ValueError(f'Prompt token IDs differ: {name}')
        hyps=[r['translation'].strip() for r in rows]
        stats=np.asarray(metric._extract_corpus_statistics(hyps,[refs]),dtype=np.int64)
        stats_by_name[name]=stats
        item={'output':identity(output),'earlier_output':identity(cold),
              'repeat_exact_matches':sum(a['translation']==b['translation'] for a,b in zip(rows,earlier)),
              'earlier_chrf_pp':metric.corpus_score([r['translation'].strip() for r in earlier],[refs]).score,
              'valid':True,'failed':0,'empty':0,'truncated':0,'groups':{}}
        if parallel==256:
            local_refs=[r['translation'].strip() for r in aligned(root/'teacher-p256/warm-1.jsonl',True)]
            item['same_concurrency_teacher']={'chrf_pp':metric.corpus_score(hyps,[local_refs]).score,
                'exact_matches':sum(a==b for a,b in zip(hyps,local_refs))}
        for group,ix in groups.items():
            item['groups'][group]={'count':len(ix),
                'chrf_pp':metric._compute_score_from_stats(stats[ix].sum(axis=0).tolist()).score,
                'exact_matches':sum(hyps[i]==refs[i] for i in ix),
                'character_edit_rate':sum(Levenshtein.distance(hyps[i],refs[i]) for i in ix)/sum(len(refs[i]) for i in ix),
                'length_ratio':sum(len(hyps[i]) for i in ix)/sum(len(refs[i]) for i in ix)}
        if not native:
            forced=aligned(folder/'teacher-forced.jsonl')
            all_scores=[]
            nll=[]
            for ref,got in zip(tf_teacher,forced):
                if [x['token'] for x in ref['scores']] != [x['token'] for x in got['scores']]:
                    raise ValueError(f'Teacher-forcing token mismatch: {name}')
                all_scores.extend(zip(ref['scores'],got['scores']))
                nll.append([sum(-s['logprob'] for s in got['scores']),len(got['scores'])])
            nll_by_name[name]=np.asarray(nll)
            total=len(all_scores)
            item['teacher_forced']={'source':identity(folder/'teacher-forced.jsonl'), 'tokens':total,
                'nll_nats_per_token':sum(-b['logprob'] for a,b in all_scores)/total,
                'raw_top1_agreement_with_teacher':sum(a['top1']==b['top1'] for a,b in all_scores)/total,
                'teacher_token_is_raw_top1':sum(b['rank']==1 for a,b in all_scores)/total,
                'teacher_token_in_top5':sum(b['rank']<=5 for a,b in all_scores)/total}
        report['strategies'][name]=item
        print(name,json.dumps(item['groups']['overall']),item.get('teacher_forced',{}).get('nll_nats_per_token'),flush=True)
    pairs=[('teacher-p32','bf16-int8-p32'),('teacher-p32','w4a16-bf16-p32'),
        ('w4a16-bf16-p32','w4a4-bf16-p32'),('w4a4-bf16-p32','w4a4-int8-p32'),
        ('llama-f16-p32','llama-q8-p32'),('llama-q8-p32','w4a4-int8-p32'),
        ('teacher-p32','w4a4-int8-p32'),('teacher-p32','teacher-p256'),
        ('teacher-p32','int4-teacher-p32'),('w4a4-int8-p32','int4-teacher-p32'),
        ('w4a4-bf16-p256','w4a4-int8-p256'),('w4a4-int8-p32','w4a4-int8-p256')]
    rng=np.random.default_rng(20260920)
    strata=[np.asarray(ix) for key,ix in groups.items() if key.startswith('target_lang:')]
    samples=[np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in strata]) for _ in range(1000)]
    def chrf(stats,ix):
        return metric._compute_score_from_stats(stats[ix].sum(axis=0).tolist()).score
    def nll(values,ix):
        sums=values[ix].sum(axis=0)
        return sums[0]/sums[1]
    for base,candidate in pairs:
        if base not in stats_by_name or candidate not in stats_by_name:
            continue  # Archived engine groups are optional in new vLLM-only runs.
        deltas=[chrf(stats_by_name[candidate],ix)-chrf(stats_by_name[base],ix) for ix in samples]
        item={'delta_chrf_pp':chrf(stats_by_name[candidate],groups['overall'])-chrf(stats_by_name[base],groups['overall']),
              'ci95_chrf_pp':np.quantile(deltas,[.025,.975]).tolist()}
        if base in nll_by_name and candidate in nll_by_name:
            deltas=[nll(nll_by_name[candidate],ix)-nll(nll_by_name[base],ix) for ix in samples]
            item['delta_nll_nats_per_token']=nll(nll_by_name[candidate],groups['overall'])-nll(nll_by_name[base],groups['overall'])
            item['ci95_nll']=np.quantile(deltas,[.025,.975]).tolist()
        report['paired_differences'][f'{candidate} minus {base}']=item
    report['chrf_signature']=str(metric.get_signature())
    if args.repeat_check:
        folder=args.repeat_check
        run_report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        original=aligned(root/'w4a4-bf16-p32/warm-1.jsonl',True)
        check={'purpose':'Investigate observed W4A4/BF16 KV free-generation repeat drift', 'phases':[]}
        first=None
        for run in run_report['runs']:
            path=folder/(run['phase']+'.jsonl')
            rows=aligned(path,True)
            if first is None:
                first=rows
            check['phases'].append({'phase':run['phase'],'output':identity(path),
                'chrf_pp':metric.corpus_score([r['translation'].strip() for r in rows],[refs]).score,
                'exact_matches_with_first':sum(a['translation']==b['translation'] for a,b in zip(rows,first)),
                'exact_matches_with_original_warm':sum(a['translation']==b['translation'] for a,b in zip(rows,original))})
        old=aligned(root/'w4a4-bf16-p32/teacher-forced.jsonl')
        new=aligned(folder/'teacher-forced.jsonl')
        pairs=[]
        for a,b in zip(old,new):
            if [s['token'] for s in a['scores']] != [s['token'] for s in b['scores']]:
                raise ValueError('Repeat-check teacher token mismatch')
            pairs.extend(zip(a['scores'],b['scores']))
        check['teacher_forced']={'source':identity(folder/'teacher-forced.jsonl'),
            'tokens':len(pairs),'nll_nats_per_token':sum(-b['logprob'] for a,b in pairs)/len(pairs),
            'max_abs_logprob_difference':max(abs(a['logprob']-b['logprob']) for a,b in pairs),
            'raw_top1_agreement_with_original_run':sum(a['top1']==b['top1'] for a,b in pairs)/len(pairs)}
        report['repeatability_check']=check
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
