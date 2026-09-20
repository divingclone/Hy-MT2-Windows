"""Release-only exclusions for the pinned, text-only HyMT runtime.

Development environments stay intact. Every release uses the same filter during
staging AND portable packaging, including --skip-stage. Do not generalize this
to arbitrary models or other vLLM/PyTorch versions without cold GPU validation.
"""
from pathlib import PurePosixPath

# Optional video/TileLang backends and Numba's optional speculative decoder.
OPTIONAL_PACKAGES = ('tilelang', 'torchcodec', 'pynvvideocodec', 'numba', 'llvmlite')
OPTIONAL_BINARIES = {
    # NVIDIA's GRAPH_JIT_ONLY configuration retains only the dispatcher, graph
    # and runtime-compiled engines. vllm_runtime.environment selects that mode.
    'torch/lib/cudnn_adv64_9.dll',
    'torch/lib/cudnn_cnn64_9.dll',
    'torch/lib/cudnn_engines_precompiled64_9.dll',
    'torch/lib/cudnn_heuristic64_9.dll',
    'torch/lib/cudnn_ops64_9.dll',
    'torch/lib/cusolverMg64_12.dll',  # multi-GPU solver, no PE consumers
    'torch/lib/nvrtc64_130_0.alt.dll',  # unused alternate compiler, keep normal NVRTC
    'vllm/_moe_C.pyd',
    'vllm/_moe_C_stable_libtorch.pyd',  # dense HyMT has no mixture-of-experts layers
    'vllm/_qutlass_C.pyd',  # CPython selects the retained cp312-win_amd64 binary
}


def excluded(relative, *, include_webview2=False):
    path=PurePosixPath(str(relative).replace('\\','/'))
    parts=path.parts
    if not include_webview2 and parts[:2]==('runtime','webview2'): return True
    if parts[:4]!=('runtime','vllm','Lib','site-packages'): return False
    rest=parts[4:]
    if not rest: return False
    name=rest[0].lower()
    if any(name==p or (name.startswith(p+'-') and name.endswith('.dist-info')) for p in OPTIONAL_PACKAGES): return True
    # CUDA Python's developer toolkit duplicates runtime DLLs already in Torch.
    # Keep headers/CCCL for imports; bundled Triton supplies the actual compiler.
    if rest[:2]==('nvidia','cu13') and len(rest)>2 and rest[2] in ('bin','lib','nvvm'): return True
    return '/'.join(rest) in OPTIONAL_BINARIES
