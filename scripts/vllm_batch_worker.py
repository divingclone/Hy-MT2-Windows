"""GPU subprocess for translate_batch.py; imports vLLM only after validation."""
import json,sys,time
from pathlib import Path
from translate import translation_prompt


def main():
    config,source,dest,summary=map(Path,sys.argv[1:5])
    limit,seed=map(int,sys.argv[5:7])
    greedy='--greedy' in sys.argv[7:]
    cfg=json.loads(config.read_text(encoding='utf-8'))['llm']
    cases=[json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    from vllm import LLM,SamplingParams
    start=time.perf_counter()
    model=LLM(**cfg)
    initialization=time.perf_counter()-start
    tokenizer=model.get_tokenizer()
    prompts=[]
    for case in cases:
        text=tokenizer.apply_chat_template([{'role':'user','content':translation_prompt(case['text'],case['target_lang'])}],tokenize=False,add_generation_prompt=True)
        tokens=tokenizer.encode(text,add_special_tokens=True)
        if len(tokens)+limit>cfg['max_model_len']:
            raise ValueError(f'Request {case["id"]}: prompt + max-tokens exceeds context; increase --context')
        prompts.append({'prompt_token_ids':tokens})
    params=[SamplingParams(temperature=0 if greedy else .7,top_p=1 if greedy else .6,
        top_k=1 if greedy else 20,repetition_penalty=1.05,max_tokens=limit,seed=seed+i) for i in range(len(cases))]
    start=time.perf_counter()
    outputs=model.generate(prompts,params,use_tqdm=False)
    wall=time.perf_counter()-start
    rows=[]
    for case,result in zip(cases,outputs):
        output=result.outputs[0]
        rows.append({'id':case['id'],'target_lang':case['target_lang'],'translation':output.text,
            'ok':bool(output.text.strip()) and output.finish_reason=='stop','truncated':output.finish_reason=='length',
            'completion_tokens':len(output.token_ids),'prompt_tokens':len(result.prompt_token_ids),'finish_reason':output.finish_reason})
    if len(rows)!=len(cases): raise RuntimeError('Incomplete batch output')
    with dest.open('x',encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    report={'backend':'vllm','requests':len(rows),'initialization_s':initialization,'wall_s':wall,
        'completion_tokens':sum(r['completion_tokens'] for r in rows),'failed':sum(not r['ok'] for r in rows),
        'truncated':sum(r['truncated'] for r in rows)}
    report['completion_tokens_per_second']=report['completion_tokens']/wall
    summary.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return 2 if report['failed'] else 0

if __name__=='__main__': raise SystemExit(main())
