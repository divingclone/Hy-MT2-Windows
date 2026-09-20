"""Native Windows KV comparison using the pretokenized engine benchmark.

Stop other GPU inference services first and dot-source vllm_windows_env.ps1.
Fresh output directory required. Does not modify or manage production services.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from vllm_runtime import environment, python_executable
from benchmark_common import ROOT, build_requests, save, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['bf16-triton', 'int8'],
                        choices=['bf16', 'fp8', 'int8', 'fp8-dynamic', 'llama-q8', 'bf16-triton'])
    parser.add_argument('--parallel', nargs='+', type=int, choices=[32, 256], default=[32, 256])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--input', type=Path, help='Default: 64 repository benchmark cases repeated eight times')
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--native', type=Path, default=ROOT/'.local/core-benchmark-v3/hy-batch-core.exe')
    parser.add_argument('--keep-going', action='store_true', help='Record failed candidates and continue screening')
    args = parser.parse_args()
    if min(args.trials, args.repeats, args.context, args.max_tokens) < 1:
        parser.error('Counts must be positive')
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    if args.input is None:
        _, _, cases = build_requests(ROOT/'scripts/benchmark_cases.json', 64, 8)
        args.input = dest/'input.jsonl'
        args.input.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in cases), encoding='utf-8')
    env = environment(ROOT)
    env.update(LLAMA_HYMT_FUSED_PROJ='1', GGML_CUDA_HYMT_DISABLE_ROPE_NORM='0',
        GGML_CUDA_HYMT_ROPE_NORM_STRICT='1', LLAMA_HYMT_SPARSE_PENALTIES='1',
        GGML_CUDA_HYMT_DISABLE_TOPK='0', LLAMA_HYMT_SAMPLING_SNAPSHOT='1',
        LLAMA_HYMT_BATCH_SNAPSHOT='1', GGML_CUDA_HYMT_EAGER_GRAPHS='1',
        GGML_CUDA_GRAPH_OPT='0', GGML_CUDA_HYMT_DISABLE_Q8_KV_FUSION='0',
        GGML_CUDA_Q8_KV_SUBWARP='0', CUDA_CACHE_PATH=str(ROOT/'cache/cuda'))
    env.update(PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='0', HYMT_VLLM_NATIVE='1', HYMT_CORE_REPEATS=str(args.repeats))
    if 'llama-q8' in args.variants:
        env['PATH'] = os.pathsep.join([str(ROOT/'archive/llama-cpp/bin'), str(ROOT/'archive/llama-cpp/runtime/cuda'), str(ROOT/'archive/llama-cpp/runtime/msvc'), env['PATH']])
    save(dest/'protocol.json', {'variants':args.variants, 'parallel':args.parallel, 'trials':args.trials,
        'repeats':args.repeats, 'context':args.context, 'max_tokens':args.max_tokens,
        'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'cache_capacity':'BF16 3/6 GiB; FP8 half; per-token-head adds FP32 scales (132/256 bytes for head_dim 128).',
        'metric':'Pretokenized core engine; first pass and subsequent warm passes separated.'})
    for trial in range(1,args.trials+1):
        for parallel in (args.parallel if trial%2 else args.parallel[::-1]):
            for variant in (args.variants if trial%2 else args.variants[::-1]):
                name=f'{variant}-p{parallel}-t{trial}'
                out=dest/name
                if variant=='llama-q8':
                    out.mkdir()
                    command=[str(args.native.resolve()), '--model',str(ROOT/'archive/llama-cpp/models/Hy-MT2-1.8B-NVFP4-fused.gguf'),
                        '--input',str(args.input.resolve()),'--output',str(out/'output'),'--summary',str(out/'summary'),
                        '--parallel',str(parallel),'--context',str(args.context),'--max-tokens',str(args.max_tokens),
                        '--batch-size','2048','--ubatch-size','2048','--threads','8','--seed','42',
                        '--sampling-threads','1','--gpu-prefix','--cache-type-k','q8_0','--cache-type-v','q8_0']
                else:
                    dtype={'bf16':'auto','bf16-triton':'auto','fp8':'fp8_e4m3',
                           'int8':'int8_per_token_head','fp8-dynamic':'fp8_per_token_head'}[variant]
                    ratio=1 if variant.startswith('bf16') else (132/256 if variant in ('int8','fp8-dynamic') else .5)
                    cfg={'model':str(ROOT/'models/Hy-MT2-1.8B-NVFP4-vllm'),'dtype':'bfloat16',
                        'max_model_len':args.context,'max_num_seqs':parallel,'max_num_batched_tokens':2048,
                        'enable_prefix_caching':False,'kv_cache_memory_bytes':int((3 if parallel==32 else 6)*1024**3*ratio),
                        'kv_cache_dtype':dtype,'disable_log_stats':True,
                        'kernel_config':{'linear_backend':'cutlass','enable_flashinfer_autotune':False}}
                    if variant in ('int8','fp8-dynamic','bf16-triton'):
                        cfg['attention_config']={'backend':'TRITON_ATTN'}
                    config=dest/(name+'.config.json')
                    save(config,cfg)
                    command=[str(python_executable(ROOT)),str(ROOT/'scripts/benchmark_vllm_offline.py'),
                        '--config',str(config),'--input',str(args.input.resolve()),'--pretokenized',
                        '--repeats',str(args.repeats),'--max-tokens',str(args.max_tokens),'--output',str(out)]
                save(dest/(name+'.command.json'),command)
                start=time.time()
                with telemetry(dest/(name+'.gpu.jsonl')), (dest/(name+'.log')).open('wb') as stream:
                    proc=subprocess.Popen(command,env=env,stdout=stream,stderr=stream,creationflags=subprocess.CREATE_NO_WINDOW)
                    while True:
                        try:
                            code=proc.wait(timeout=3)
                            break
                        except subprocess.TimeoutExpired:
                            log_text=(dest/(name+'.log')).read_text(encoding='utf-8',errors='replace')
                            if 'EngineCore failed to start.' in log_text or time.time()-start > 1200:
                                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                                proc.wait(timeout=15)
                                code=-1
                                break
                save(dest/(name+'.process.json'),{'exit_code':code,'process_wall_s':time.time()-start})
                print(name,'exit',code,flush=True)
                if code:
                    if args.keep_going:
                        continue
                    raise RuntimeError(f'{name} failed; inspect log')
                if variant!='llama-q8':
                    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
                    print(report['warm_median'],flush=True)


if __name__=='__main__':
    main()
