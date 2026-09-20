"""Verify the custom CUDA DLL before optionally copying an isolated candidate.

Requires pefile in the build environment; never changes the source runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re
import subprocess

import pefile

# PyTorch exceeds pefile's conservative malware-analysis defaults (8192).
# Both IAT and lookup entries count toward its import limit.
pefile.MAX_IMPORT_SYMBOLS = 1_000_000

OPTIONAL = {'cufft64_12.dll', 'cusparse64_12.dll', 'cusolver64_12.dll'}
# nvJitLink's native users are cuFFT/cuSPARSE. The CUDA Python bindings retain
# an explicit optional nvjitlink API, unused by this text inference profile.
REMOVABLE = OPTIONAL | {'nvjitlink_130_0.dll', 'cufftw64_12.dll'}
TORCH_LIB = Path('runtime/vllm/Lib/site-packages/torch/lib')
COMMIT = '70d99e998b4955e0049d13a98d77ae1b14db1f45'
CUDA_PYTHON_SHA256 = 'a594665c3bacf798015c95c83f68c485e36f9f17ecb98dfdf88820a9066de6bd'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def imports(path):
    with pefile.PE(str(path), fast_load=True, max_symbol_exports=1_000_000) as pe:
        pe.parse_data_directories(directories=[1, 13])
        def table(attribute):
            return {
                entry.dll.decode('ascii').lower(): {
                    item.name.decode('ascii') if item.name else item.ordinal
                    for item in entry.imports
                }
                for entry in getattr(pe, attribute, [])
            }
        return table('DIRECTORY_ENTRY_IMPORT'), table('DIRECTORY_ENTRY_DELAY_IMPORT')


def exports(path):
    with pefile.PE(str(path), fast_load=True, max_symbol_exports=1_000_000) as pe:
        pe.parse_data_directories(directories=[0])
        symbols = set()
        for item in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            symbols.add(item.ordinal)
            if item.name:
                symbols.add(item.name.decode('ascii'))
        return symbols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True, help='An already trimmed payload root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=Path, help='Create a NEW isolated candidate payload directory')
    parser.add_argument('--cuobjdump', default=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\cuobjdump.exe')
    args = parser.parse_args()
    base = args.base.resolve()
    dll = args.dll.resolve()
    provenance = json.loads(dll.with_name('hymt-build.json').read_text(encoding='utf-8-sig'))
    if provenance['source_commit'] != COMMIT or provenance['custom_dll_sha256'] != digest(dll):
        raise SystemExit('Missing or mismatched custom build provenance')
    for name, expected in provenance['base_libraries'].items():
        if digest(base / TORCH_LIB / name) != expected:
            raise SystemExit(f'The candidate base has a different {name}')
    elf_list = subprocess.check_output([args.cuobjdump, '--list-elf', str(dll)], text=True)
    architectures = sorted(set(re.findall(r'\bsm_\d+[af]?\b', elf_list)),
                           key=lambda s: (int(re.search(r'\d+', s)[0]), s))
    # Some upstream per-operator CMake rules add optimized architectures beyond
    # TORCH_CUDA_ARCH_LIST. Keep those kernels; reject unrelated wheel defaults.
    required_architectures = {'sm_80', 'sm_86', 'sm_120'}
    allowed_architectures = required_architectures | {'sm_89', 'sm_120a', 'sm_121a'}
    if not required_architectures <= set(architectures) <= allowed_architectures:
        raise SystemExit(f'Unexpected CUDA machine code architectures: {architectures}')
    cuda_python = base / TORCH_LIB.parent / 'cuda/__init__.py'
    if digest(cuda_python) != CUDA_PYTHON_SHA256:
        raise SystemExit('Unexpected original torch.cuda source')
    normal, delayed = imports(dll)
    problems = []
    if REMOVABLE & normal.keys():
        problems.append('Optional libraries still have mandatory imports')
    if not OPTIONAL <= delayed.keys():
        problems.append('Expected delay imports were not generated')
    new_exports = exports(dll)
    consumers = []
    native_count = 0
    for path in base.rglob('*'):
        if path.suffix.lower() not in {'.pyd', '.dll', '.exe'} or not path.is_file():
            continue
        relative = path.relative_to(base)
        if relative.parent == TORCH_LIB and path.name.lower() in REMOVABLE | {'torch_cuda.dll'}:
            continue
        direct, lazy = imports(path)
        native_count += 1
        for dependency in REMOVABLE & direct.keys():
            problems.append(f'{relative} still directly needs {dependency}')
        required = direct.get('torch_cuda.dll', set()) | lazy.get('torch_cuda.dll', set())
        if required:
            if any(isinstance(symbol, int) for symbol in required):
                problems.append(f'{relative}: ordinal CUDA imports require an explicit ABI mapping')
            missing = required - new_exports
            consumers.append({'path': relative.as_posix(), 'symbols': len(required)})
            if missing:
                problems.append(f'{relative}: missing CUDA exports {sorted(map(str, missing))}')
    for name in ('torch_cpu.dll', 'c10.dll', 'c10_cuda.dll'):
        missing = normal.get(name, set()) - exports(base / TORCH_LIB / name)
        if missing:
            problems.append(f'Original {name} lacks {sorted(map(str, missing))}')
    report = {
        'schema': 1, 'profile': 'hymt-cuda-delay-v1', 'source_commit': COMMIT,
        'build_provenance': provenance, 'architectures': architectures,
        'custom_dll_sha256': digest(dll), 'custom_dll_bytes': dll.stat().st_size,
        'original_dll_sha256': digest(base / TORCH_LIB / 'torch_cuda.dll'),
        'original_dll_bytes': (base / TORCH_LIB / 'torch_cuda.dll').stat().st_size,
        'mandatory_imports': sorted(normal), 'delay_imports': sorted(delayed),
        'removed': sorted(REMOVABLE), 'native_files_checked': native_count,
        'consumers': consumers, 'problems': problems,
        'binary_compatibility_checked': not problems,
        'inference_validated': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if problems:
        raise SystemExit('\n'.join(problem[:1000] for problem in problems))
    if args.stage:
        stage = args.stage.resolve()
        if stage == base or base in stage.parents or stage in base.parents:
            raise SystemExit('The candidate must be separate from the base runtime')
        def ignore(folder, names):
            if Path(folder).resolve() == base / TORCH_LIB:
                return {name for name in names if name.lower() in REMOVABLE | {'torch_cuda.dll'}}
            return {'__pycache__'} & set(names)
        shutil.copytree(base, stage, ignore=ignore)
        shutil.copy2(dll, stage / TORCH_LIB / 'torch_cuda.dll')
        # The unchanged native Python binding embeds the official wheel's old
        # architecture list. Make the public query describe the actual new DLL.
        source = cuda_python.read_bytes().decode('utf-8')
        before = '    arch_flags = torch._C._cuda_getArchFlags()'
        if source.count(before) != 1:
            raise SystemExit('Unexpected torch.cuda architecture query')
        after = f'    arch_flags = {" ".join(architectures)!r}  # HyMT custom CUDA backend'
        patched = stage / TORCH_LIB.parent / 'cuda/__init__.py'
        patched.write_bytes(source.replace(before, after).encode('utf-8'))
        report['cuda_python_sha256'] = digest(patched)
        (stage / TORCH_LIB.parent / 'hymt-build.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        runtime_path = stage / 'runtime/vllm/hymt-runtime.json'
        runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
        runtime['torch_custom_build'] = report['profile']
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding='utf-8')
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
