"""Shared native-Windows vLLM runtime, environment and command construction."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def python_executable(root=ROOT):
    path=Path(root)/'runtime/vllm/python.exe'
    if not path.is_file():
        raise ValueError('缺少 vLLM 运行环境。请运行 setup-runtime.cmd，或使用完整 vLLM 运行包。')
    return path


def environment(root=ROOT, *, cache=None, gpu='0'):
    root=Path(root)
    removed=('LLAMA_', 'GGML_', 'CUDA', 'VC', 'VS', 'WINDOWSSDK', 'UCRT',
             'UNIVERSALCRT', 'INCLUDE', 'LIB', 'CMAKE', 'PYTHON', 'TORCH', 'VLLM', 'TRITON', 'FLASHINFER')
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(removed)}
    env.update(PYTHONUTF8='1', PYTHONNOUSERSITE='1', HYMT_VLLM_NATIVE='1',
        VLLM_HOST_IP='127.0.0.1', TOKENIZERS_PARALLELISM='false', VSLANG='1033',
        CUDA_VISIBLE_DEVICES=str(gpu), MAX_JOBS='8')
    runtime=python_executable(root).parent
    packages=runtime/'Lib/site-packages'
    cuda=packages/'triton/backends/nvidia'
    compiler=packages/'triton/runtime/tcc/tcc.exe'
    for path in (compiler,cuda/'bin/ptxas.exe',cuda/'bin/cudart64_13.dll',
                 packages/'flashinfer/data/aot/sampling/sampling.dll'):
        if not path.is_file():
            raise ValueError(f'免安装运行包不完整，缺少 {path.name}；请使用完整运行包或按 BUILD 文档重新构建。')
    system=Path(env.get('SystemRoot','C:/Windows'))/'System32'
    env['PATH']=os.pathsep.join(map(str,[runtime,runtime/'Scripts',packages/'torch/lib',system,system/'WindowsPowerShell/v1.0']))
    # Triton ships TinyCC + PTXAS. FlashInfer uses our fat AOT DLL, never NVCC.
    env.update(CC=str(compiler), CXX=str(runtime/'no-host-cpp-compiler.exe'),
               CUDA_PATH=str(cuda), CUDA_HOME=str(cuda), FLASHINFER_DISABLE_JIT='1',
               CUDNN_LIB_CONFIG='GRAPH_JIT_ONLY',
               VLLM_USE_FLASHINFER_SAMPLER='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
    cache=Path(cache or root/'cache/vllm').resolve()
    cache.mkdir(parents=True,exist_ok=True)
    env.update(VLLM_CACHE_ROOT=str(cache/'vllm'), TRITON_CACHE_DIR=str(cache/'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), FLASHINFER_CACHE_DIR=str(cache/'flashinfer'),
               CUDA_CACHE_PATH=str(cache/'cuda'), HF_HOME=str(cache/'huggingface'))
    return env


def llm_config(plan):
    # vLLM 0.29 defaults to 2 * max_num_seqs captured tokens (64 at 32
    # sequences). Mixed prefill + decode frequently exceeds that limit even
    # at low request concurrency, falling back to uncaptured execution.
    # Keep the normal 512 ceiling at 256 sequences, and respect smaller
    # scheduler budgets. This does not increase the active sequence limit.
    graph_tokens=min(plan['batch'],max(256,min(2*plan['parallel'],512)))
    return {'model':plan['model'],'dtype':'bfloat16','max_model_len':plan['context'],
        'max_num_seqs':plan['parallel'],'max_num_batched_tokens':plan['batch'],
        'kv_cache_memory_bytes':int(plan['budget']['kv_total_mib']*1024**2),
        'kv_cache_dtype':plan['cache'],'enable_prefix_caching':False,'disable_log_stats':True,
        'compilation_config':{'max_cudagraph_capture_size':graph_tokens},
        'attention_config':{'backend':'TRITON_ATTN'},
        'kernel_config':{'linear_backend':plan['kernel'],'enable_flashinfer_autotune':False}}


def server_command(plan, *, root=ROOT, port=18080):
    config=llm_config(plan)
    command=[str(python_executable(root)),'-u','-s','-X','utf8','-m','vllm.entrypoints.cli.main',
        'serve',config.pop('model'),'--host','127.0.0.1','--port',str(port),
        '--served-model-name','hy-mt2','--generation-config','vllm',
        '--override-generation-config',json.dumps({'temperature':.7,'top_p':.6,'top_k':20,'repetition_penalty':1.05}),
        '--disable-uvicorn-access-log','--stream-interval','1']
    for key,value in config.items():
        if isinstance(value,bool):
            command.append('--'+('' if value else 'no-')+key.replace('_','-'))
        else:
            command.extend(['--'+key.replace('_','-'),json.dumps(value) if isinstance(value,dict) else str(value)])
    return command


def stop_process_tree(process):
    if process.poll() is None:
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    process.wait(timeout=30)
