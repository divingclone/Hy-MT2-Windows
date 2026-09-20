"""Release-only exclusions for the pinned, text-only HyMT runtime.

Development environments stay intact. Every release uses the same filter during
staging AND portable packaging, including --skip-stage. Do not generalize this
to arbitrary models or other vLLM/PyTorch versions without cold GPU validation.
"""
from pathlib import PurePosixPath

# Optional multimedia/backends, speculative decoding and Dynamo's Z3 debugger.
OPTIONAL_PACKAGES = ('tilelang', 'torchcodec', 'pynvvideocodec', 'numba', 'llvmlite',
                     'cv2', 'opencv_python_headless', 'torchaudio', 'z3', 'z3_solver')
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
    # No retained PE imports or dynamic-loader references to the host cuRAND
    # API. Device-side random kernels are already compiled into Torch/sampling.
    'torch/lib/curand64_10.dll',
    'torch/lib/nvperf_host.dll',  # optional CUPTI metrics profiler, not inference
    'vllm/_moe_C.pyd',
    'vllm/_moe_C_stable_libtorch.pyd',  # dense HyMT has no mixture-of-experts layers
    'vllm/_qutlass_C.pyd',  # CPython selects the retained cp312-win_amd64 binary
    # Text inference selects TRITON_ATTN. runtime_patches defers the unrelated
    # vision module's eager import; do not remove this binary without that patch.
    'vllm/vllm_flash_attn/_vllm_fa2_C.pyd',
}

# Explicitly scoped developer assets. Do not remove all folders named "testing":
# torch/numpy/sympy import parts of their testing helpers at runtime.
TEST_PACKAGES = frozenset(('numpy', 'sympy', 'networkx', 'mpmath', 'PIL',
                           'setuptools', 'pip', 'psutil', 'zmq'))
DEVELOPER_TREES = (
    ('samples',),  # orphaned PyNvVideoCodec examples (wheel RECORD verified)
    ('pycountry', 'locales'),  # gettext catalogs; retain ISO language databases
)
PYTHON_DEVELOPER_TREES = (
    ('tcl',), ('Lib', 'tkinter'), ('Lib', 'idlelib'), ('Lib', 'turtledemo'),
    ('Lib', 'ensurepip'),
)
TK_BINARIES = frozenset(('_tkinter.pyd', 'tcl86t.dll', 'tk86t.dll'))
AOT_SOURCE_TREES = frozenset(('cccl', 'csrc', 'cutlass', 'include', 'spdlog'))
SOURCE_SUFFIXES = frozenset(('.h', '.hpp', '.cuh', '.cu', '.c', '.cc', '.cpp', '.inl', '.ipp'))


def exclusion_reason(relative, *, include_webview2=False):
    """Return a stable audit category, or None for a retained file/directory."""
    path=PurePosixPath(str(relative).replace('\\','/'))
    parts=path.parts
    if not include_webview2 and parts[:2]==('runtime','webview2'): return 'webview-on-demand'
    if len(parts)>=2 and parts[0]=='runtime' and parts[1] in ('python','vllm'):
        local=parts[2:]
        if any(local[:len(prefix)]==prefix for prefix in PYTHON_DEVELOPER_TREES):
            return 'python-developer-tools'
        if len(local)==2 and local[0]=='DLLs' and local[1] in TK_BINARIES:
            return 'python-developer-tools'
    if parts[:4]!=('runtime','vllm','Lib','site-packages'): return None
    rest=parts[4:]
    if not rest: return None
    name=rest[0].lower()
    if any(name==p or (name.startswith(p+'-') and name.endswith('.dist-info')) for p in OPTIONAL_PACKAGES): return 'optional-package'
    # CUDA Python's developer toolkit duplicates runtime DLLs already in Torch.
    # Keep headers/CCCL for imports; bundled Triton supplies the actual compiler.
    if rest[:2]==('nvidia','cu13') and len(rest)>2 and rest[2] in ('bin','lib','nvvm'): return 'duplicate-cuda-toolkit'
    if '/'.join(rest) in OPTIONAL_BINARIES: return 'optional-binary'
    if any(rest[:len(prefix)]==prefix for prefix in DEVELOPER_TREES): return 'developer-assets'
    if rest==('PyWin32.chm',): return 'developer-assets'
    # Sampling loads its bundled AOT DLL before considering JIT sources. Keep
    # license/notice files and Python loaders, and keep Triton's separate SDK.
    if (rest[:2]==('flashinfer','data') and len(rest)>3 and rest[2] in AOT_SOURCE_TREES
            and path.suffix.lower() in SOURCE_SUFFIXES
            and not path.name.upper().startswith(('LICENSE','COPYING','NOTICE'))):
        return 'aot-build-sources'
    if rest[:2]==('mistral_common','data') and len(rest)==3 and rest[2].startswith(
            ('mistral_instruct_tokenizer_', 'tekken_', 'tokenizer.model.')):
        return 'other-model-tokenizers'
    if rest[0] in TEST_PACKAGES and 'tests' in rest[1:]: return 'package-tests'
    # The supported runtime uses TinyCC/PTXAS and prebuilt extensions, not MSVC
    # linking. Keep Torch headers, retained DLLs, Python/Triton/CUDA import libraries.
    if rest[:2]==('torch','lib') and path.suffix.lower() in ('.lib','.exp'):
        return 'torch-link-libraries'
    return None


def excluded(relative, *, include_webview2=False):
    return exclusion_reason(relative, include_webview2=include_webview2) is not None
