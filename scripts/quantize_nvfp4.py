"""Build an opt-in NVFP4 candidate from high-precision Hy-MT2 GGUF weights.

Requires the patched llama-quantize with --nvfp4-mse. An optional importance
matrix must be collected from the unpacked high-precision model, before fusion.
Outputs are candidates: measure held-out quality and speed before deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path: Path) -> dict:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'filename': path.name, 'size_bytes': path.stat().st_size, 'sha256': digest}


def inspect_source(path: Path) -> dict:
    sys.path.insert(0, str(ROOT / 'src/llama.cpp/gguf-py'))
    import gguf
    reader = gguf.GGUFReader(path)
    expected = {'general.architecture': 'hunyuan-dense',
                'hunyuan-dense.block_count': 32, 'hunyuan-dense.embedding_length': 2048,
                'hunyuan-dense.feed_forward_length': 6144,
                'hunyuan-dense.attention.head_count': 16,
                'hunyuan-dense.attention.head_count_kv': 4}
    for key, value in expected.items():
        if key not in reader.fields or reader.fields[key].contents() != value:
            raise ValueError(f'Expected high-precision Hy-MT2-1.8B: {key}')
    if 'hunyuan-dense.hymt_packed_projections' in reader.fields:
        raise ValueError('Use the original unpacked high-precision model, before projection fusion')
    if 'split.count' in reader.fields and reader.fields['split.count'].contents() != 1:
        raise ValueError('Merge GGUF shards before quantization')
    allowed = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
               gguf.GGMLQuantizationType.BF16}
    if not reader.tensors or any(t.tensor_type not in allowed for t in reader.tensors):
        raise ValueError('Source contains low-bit weights; requantization is not supported')
    return {'tensors': len(reader.tensors), 'types': sorted({t.tensor_type.name for t in reader.tensors}),
            'matrix_input_sizes': {t.name: int(t.shape[0]) for t in reader.tensors
                                   if len(t.shape) == 2 and t.name.endswith('.weight')}}


def inspect_imatrix(path: Path, source_info: dict) -> dict:
    """Check current GGUF imatrix structure and coverage; it has no checkpoint hash."""
    import gguf
    import numpy as np
    reader = gguf.GGUFReader(path)
    keys = ('imatrix.datasets', 'imatrix.chunk_count', 'imatrix.chunk_size')
    if any(key not in reader.fields for key in keys):
        raise ValueError('Use a GGUF imatrix with dataset, chunk count and chunk size metadata')
    chunks, context = (int(reader.fields[key].contents()) for key in keys[1:])
    if chunks < 1 or context < 1:
        raise ValueError('Imatrix contains no completed calibration chunks')
    tensors = {t.name: t for t in reader.tensors}
    names = {name.removesuffix(suffix) for name in tensors
             for suffix in ('.in_sum2', '.counts') if name.endswith(suffix)}
    matched = []
    for name in sorted(names):
        sums, counts = tensors.get(name + '.in_sum2'), tensors.get(name + '.counts')
        if sums is None or counts is None or any(t.tensor_type != gguf.GGMLQuantizationType.F32 for t in (sums, counts)):
            raise ValueError(f'Imatrix requires paired F32 sums/counts: {name}')
        if (not np.isfinite(sums.data).all() or not np.isfinite(counts.data).all()
                or (sums.data < 0).any() or (counts.data < 1).any()):
            raise ValueError(f'Imatrix has invalid or unobserved activations: {name}')
        expected = source_info['matrix_input_sizes'].get(name)
        if expected is None or name == 'token_embd.weight':
            continue  # Embedding statistics use a different layout in llama.cpp.
        if counts.data.size != 1 or sums.data.size != expected:
            raise ValueError(f'Imatrix shape does not match source: {name}')
        if not (sums.data > 0).any():
            raise ValueError(f'Imatrix has no positive importance weights: {name}')
        matched.append(name)
    if not matched:
        raise ValueError('Imatrix has no compatible weight entries for this source')
    missing = sorted(set(source_info['matrix_input_sizes']) - set(matched))
    return {'chunks': chunks, 'context': context, 'matched_tensors': matched,
            'unweighted_tensors': missing,
            'note': 'Matching names/shapes do not establish the originating checkpoint or held-out separation.'}


def require_help(executable: Path, flags: tuple[str, ...], env: dict) -> None:
    if not executable.is_file():
        raise FileNotFoundError(f'Missing executable: {executable}; rebuild the patched source (docs/BUILD.md)')
    result = subprocess.run([str(executable), '--help'], env=env, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=30,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    # llama-quantize's normal usage/help function exits with 1.
    if result.returncode not in (0, 1) or any(flag not in result.stdout + result.stderr for flag in flags):
        raise ValueError(f'{executable.name} lacks required options {flags}; rebuild the patched source (docs/BUILD.md)')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New fused candidate GGUF')
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'bin', help='Installed quantization tools; default: bin')
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--imatrix', type=Path, help='Existing GGUF imatrix from this unpacked model')
    calibration.add_argument('--calibration', type=Path, help='Representative UTF-8 text, disjoint from evaluation')
    parser.add_argument('--context', type=int, default=512, help='Calibration chunk size')
    parser.add_argument('--chunks', type=int, default=32, help='Maximum calibration chunks')
    parser.add_argument('--parse-special', action='store_true', help='Parse chat-template special tokens in calibration text')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--scale-search', choices=('local', 'full'), default='local',
                        help='local: nearby FP8 scales; full: exact bounded search of all legal scales (slower offline)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.context, args.chunks, args.threads) < 1:
        parser.error('context, chunks and threads must be positive')
    if args.parse_special and not args.calibration:
        parser.error('--parse-special requires --calibration')
    source, output, binary = args.source.resolve(), args.output.resolve(), args.binary_dir.resolve()
    if output.suffix.lower() != '.gguf':
        parser.error('output must have a .gguf extension')
    intermediate = output.with_name(output.stem + '.unfused.gguf')
    report = output.with_suffix('.quantization.json')
    log_path = output.with_suffix('.quantization.log')
    matrix = args.imatrix.resolve() if args.imatrix else output.with_suffix('.imatrix.gguf') if args.calibration else None
    outputs = [output, intermediate, report, log_path, output.with_suffix(output.suffix + '.tmp')]
    if args.calibration:
        outputs.append(matrix)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite: {path}')
    for path in [source, *([args.imatrix] if args.imatrix else []), *([args.calibration] if args.calibration else [])]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Input missing or empty: {path}')
    source_info = inspect_source(source)
    matrix_info = inspect_imatrix(matrix, source_info) if args.imatrix else None
    quantizer = binary / 'llama-quantize.exe'
    quantization_flag = '--nvfp4-mse-full' if args.scale_search == 'full' else '--nvfp4-mse'
    method = 'nvfp4-mse-full-v2' if args.scale_search == 'full' else 'nvfp4-mse-v1'
    env = os.environ.copy()
    # The explicit CLI selection is the only authority for the recorded method.
    env.pop('GGML_NVFP4_QUANTIZE_MSE', None)
    env['PATH'] = os.pathsep.join([str(binary), str(ROOT / 'runtime/cuda'), str(ROOT / 'runtime/msvc'), env.get('PATH', '')])
    require_help(quantizer, (quantization_flag, '--tensor-type', '--override-kv', '--imatrix'), env)
    commands = []
    if args.calibration:
        require_help(binary / 'llama-imatrix.exe', ('--output-format', '--no-ppl', '--chunks'), env)
        commands.append([str(binary / 'llama-imatrix.exe'), '-m', str(source),
                         '-f', str(args.calibration.resolve()), '-o', str(matrix),
                         '--output-format', 'gguf', '-c', str(args.context), '-b', str(args.context),
                         '-ub', str(args.context), '-np', '1', '--chunks', str(args.chunks), '-ngl', '99',
                         '-t', str(args.threads), '--no-ppl'])
        if args.parse_special:
            commands[0].append('--parse-special')
    command = [str(quantizer), quantization_flag, '--tensor-type', '.*=nvfp4',
               '--override-kv', f'hymt.quantization.method=str:{method}',
               '--override-kv', 'general.quantized_by=str:Hy-MT2-Windows']
    if matrix:
        command += ['--imatrix', str(matrix)]
    command += [str(source), str(intermediate), 'Q4_0', str(args.threads)]
    commands.append(command)
    plan = {'source': str(source), 'source_layout': source_info, 'output': str(output),
            'method': method, 'scale_search': args.scale_search,
            'objective': 'per-block input-column-diagonal weighted squared error' if matrix else 'per-block squared error',
            'weighted': matrix is not None, 'commands': commands,
            'imatrix_validation': matrix_info,
            'imatrix_source': 'generated_from_source' if args.calibration else 'user_supplied_unverified' if matrix else None,
            'note': 'Q4_0 is the container label; tensor overrides select NVFP4. Quality is not established by weight MSE.'}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.update(started_at_utc=datetime.now(timezone.utc).isoformat(), source_fingerprint=fingerprint(source),
                quantizer=fingerprint(quantizer), status='running')
    # The implementation lives in DLLs, so record them as well as the small launcher.
    plan['libraries'] = [fingerprint(path) for path in sorted(binary.glob('*.dll'))]
    if args.calibration:
        plan['calibration'] = fingerprint(args.calibration)
    if args.imatrix:
        plan['imatrix'] = fingerprint(matrix)
    with report.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    try:
        with log_path.open('xb') as log:
            for command in commands:
                target = matrix if args.calibration and command is commands[0] else intermediate
                if target.exists():
                    raise FileExistsError(f'Refusing to overwrite: {target}')
                subprocess.run(command, env=env, stdout=log, stderr=log, check=True,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if args.calibration and command is commands[0]:
                    plan['imatrix_validation'] = inspect_imatrix(matrix, source_info)
                    plan['imatrix'] = fingerprint(matrix)
        import gguf
        candidate = gguf.GGUFReader(intermediate)
        for key, expected in (('hymt.quantization.method', method),
                              ('general.quantized_by', 'Hy-MT2-Windows')):
            if key not in candidate.fields or candidate.fields[key].contents() != expected:
                raise ValueError(f'Quantizer did not write expected provenance: {key}')
        type_counts = {}
        for tensor in candidate.tensors:
            name = tensor.tensor_type.name
            type_counts[name] = type_counts.get(name, 0) + 1
        if not type_counts.get('NVFP4'):
            raise ValueError('Quantizer produced no NVFP4 tensors')
        plan['unfused_tensor_types'] = type_counts
        plan['unfused_fingerprint'] = fingerprint(intermediate)
        del candidate
        from repack_hymt_gguf import repack
        plan['repack'] = repack(intermediate, output)
        if matrix:
            plan['imatrix'] = fingerprint(matrix)
        plan.update(status='complete', output_fingerprint=fingerprint(output))
    except Exception as error:
        plan.update(status='failed', error=str(error))
        raise
    finally:
        report.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Candidate: {output}\nProvenance: {report}\nValidate held-out translations and throughput before selecting it.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)
