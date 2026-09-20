"""Build portable FlashInfer sampling kernels on the maintainer's Windows host.

Run after scripts/vllm_windows_env.ps1. MSVC/CUDA are build dependencies only.
The resulting AOT DLL is loaded by FlashInfer without invoking ninja or NVCC.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    os.environ['FLASHINFER_CUDA_ARCH_LIST'] = '8.0 8.6 8.9 12.0f'
    os.environ['FLASHINFER_CACHE_DIR'] = str(args.cache.resolve())
    os.environ.pop('FLASHINFER_DISABLE_JIT', None)
    from flashinfer.jit.sampling import gen_sampling_module
    spec = gen_sampling_module()
    spec.build(verbose=True)
    destination = runtime/'Lib/site-packages/flashinfer/data/aot/sampling/sampling.dll'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.jit_library_path, destination)
    spec.load(destination)
    nvcc = Path(os.environ['CUDA_PATH'])/'bin/nvcc.exe'
    evidence = {
        'module': 'sampling', 'architectures': ['sm_80', 'sm_86', 'sm_89', 'sm_120f'],
        'flashinfer': importlib.metadata.version('flashinfer-python'),
        'torch': importlib.metadata.version('torch'),
        'nvcc': subprocess.check_output([str(nvcc), '--version'], text=True).strip(),
        'size_bytes': destination.stat().st_size,
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'runtime_jit_required': False,
    }
    destination.with_suffix('.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
