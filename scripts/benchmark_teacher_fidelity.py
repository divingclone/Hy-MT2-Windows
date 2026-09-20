"""Native Windows teacher-relative quantization ablation (no HTTP).

Dot-source vllm_windows_env.ps1 and stop competing GPU services first.
This runner does not manage production services. Requires an empty output path.
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
from benchmark_common import ROOT, save, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--native', type=Path, help='Optional archived llama.cpp comparison executable')
    args = parser.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    env = environment(ROOT)
    env.update(PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='0', HYMT_VLLM_NATIVE='1', HYMT_CORE_REPEATS='1',
        LLAMA_HYMT_FUSED_PROJ='1', GGML_CUDA_HYMT_DISABLE_ROPE_NORM='0',
        GGML_CUDA_HYMT_ROPE_NORM_STRICT='1', LLAMA_HYMT_SPARSE_PENALTIES='1',
        GGML_CUDA_HYMT_DISABLE_TOPK='0', LLAMA_HYMT_SAMPLING_SNAPSHOT='1',
        LLAMA_HYMT_BATCH_SNAPSHOT='1', GGML_CUDA_HYMT_EAGER_GRAPHS='1', GGML_CUDA_GRAPH_OPT='0',
        GGML_CUDA_HYMT_DISABLE_Q8_KV_FUSION='0', GGML_CUDA_Q8_KV_SUBWARP='0',
        CUDA_CACHE_PATH=str(ROOT/'cache/cuda'))
    if args.native is not None:
        env['PATH'] = os.pathsep.join([str(ROOT/'archive/llama-cpp/bin'), str(ROOT/'archive/llama-cpp/runtime/cuda'), str(ROOT/'archive/llama-cpp/runtime/msvc'), env['PATH']])
    variants = [('teacher',32), ('bf16-int8',32), ('w4a16-bf16',32), ('w4a4-bf16',32),
                ('w4a4-int8',32), ('llama-f16',32), ('llama-q8',32),
                ('teacher',256), ('w4a4-bf16',256), ('w4a4-int8',256)]
    if args.native is None:
        variants = [(v,p) for v,p in variants if not v.startswith('llama-')]
    teacher = dest/'teacher-p32/warm-1.jsonl'
    save(dest/'protocol.json', {'variants': variants, 'input': str(args.input.resolve()),
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'reference': 'Original BF16 weights + BF16 KV, Triton attention, greedy p32 warm pass',
        'sampling': {'temperature':0, 'top_p':1, 'top_k':1, 'repetition_penalty':1.05},
        'context':4096, 'max_tokens':2048, 'free_generation_passes':2,
        'teacher_forcing':'Raw logits on original teacher prefixes; includes EOS; not a throughput benchmark',
        'attention':'All vLLM groups use TRITON_ATTN to isolate KV/linear precision',
        'environment': {k:v for k,v in env.items() if k.startswith(('LLAMA_', 'GGML_', 'HYMT_'))}})
    for variant, parallel in variants:
        name=f'{variant}-p{parallel}'
        out=dest/name
        if variant.startswith('llama-'):
            out.mkdir()
            cache='f16' if variant=='llama-f16' else 'q8_0'
            command=[str(args.native.resolve()), '--model',str(ROOT/'archive/llama-cpp/models/Hy-MT2-1.8B-NVFP4-fused.gguf'),
                '--input',str(args.input.resolve()), '--output',str(out/'output'), '--summary',str(out/'summary'),
                '--parallel',str(parallel),'--context','4096','--max-tokens','2048', '--greedy',
                '--batch-size','2048','--ubatch-size','2048','--threads','8','--seed','42',
                '--sampling-threads','1','--gpu-prefix','--cache-type-k',cache,'--cache-type-v',cache]
        else:
            original=variant in ('teacher','bf16-int8')
            int8=variant.endswith('-int8')
            cfg={'model':str(ROOT/('models/Hy-MT2-1.8B' if original else 'models/Hy-MT2-1.8B-NVFP4-vllm')),
                'dtype':'bfloat16','max_model_len':4096,'max_num_seqs':parallel,'max_num_batched_tokens':2048,
                'enable_prefix_caching':False,'disable_log_stats':True,
                'kv_cache_memory_bytes':int((3 if parallel==32 else 8)*1024**3*(132/256 if int8 else 1)),
                'kv_cache_dtype':'int8_per_token_head' if int8 else 'auto',
                'attention_config':{'backend':'TRITON_ATTN'}}
            if not original:
                cfg['kernel_config']={'linear_backend':'marlin' if variant.startswith('w4a16') else 'cutlass',
                                      'enable_flashinfer_autotune':False}
            config=dest/(name+'.config.json')
            save(config,cfg)
            command=[str(python_executable(ROOT)),str(ROOT/'scripts/benchmark_vllm_offline.py'),
                '--config',str(config),'--input',str(args.input.resolve()),'--pretokenized','--greedy',
                '--save-token-ids','--teacher-output','self' if (variant,parallel)==('teacher',32) else str(teacher),
                '--repeats','1','--max-tokens','2048','--output',str(out)]
        save(dest/(name+'.command.json'),command)
        print('Starting',name,flush=True)
        started=time.time()
        with telemetry(dest/(name+'.gpu.jsonl')), (dest/(name+'.log')).open('wb') as stream:
            proc=subprocess.Popen(command,env=env,stdout=stream,stderr=stream,creationflags=subprocess.CREATE_NO_WINDOW)
            while True:
                try:
                    code=proc.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    log=(dest/(name+'.log')).read_text(encoding='utf-8',errors='replace')
                    if 'EngineCore failed to start.' in log or time.time()-started>1200:
                        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                        proc.wait(timeout=15)
                        code=-1
                        break
        save(dest/(name+'.process.json'),{'exit_code':code,'wall_s':time.time()-started})
        print('Finished',name,'exit',code,flush=True)
        if code:
            raise RuntimeError(f'{name} failed; inspect saved log')


if __name__=='__main__':
    main()
