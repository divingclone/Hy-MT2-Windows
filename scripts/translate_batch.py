"""Translate a UTF-8 JSONL file with the specialized native batch engine."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_config import CACHE_TYPE_BYTES, cache_type_args, ensure_cache_type_support, resolve_config

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'fast': 'Hy-MT2-1.8B-NVFP4-fused.gguf',
    'official': 'Hy-MT2-1.8B-Q4_K_M-fused.gguf',
}


def check_distinct_paths(paths):
    items = list(paths.items())
    for index, (name, path) in enumerate(items):
        for other_name, other in items[:index]:
            if path == other or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError(f'{name} and {other_name} must be different files: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSONL: id, text, target_lang')
    parser.add_argument('output', type=Path, nargs='?', help='Default: INPUT.translated.jsonl')
    parser.add_argument('--profile', choices=('auto', *PROFILES), default='auto')
    parser.add_argument('--model', type=Path, help='Use an explicitly selected existing Hy-MT2-1.8B GGUF')
    parser.add_argument('--parallel', type=int, default=None, help='Default: adapt to free GPU memory')
    parser.add_argument('--gpu', help='NVIDIA GPU index or UUID; default: supported GPU with most free memory')
    parser.add_argument('--binary-dir', type=Path, help='Override the executable directory')
    parser.add_argument('--sampling-threads', type=int, choices=(1, 2, 4, 8), default=None,
                        help='Default: 1 with GPU prefix; 4 with --cpu-sampling')
    parser.add_argument('--cpu-sampling', action='store_true', help='Use the original CPU sampling path')
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--ubatch', type=int, default=None)
    parser.add_argument('--cache-type-k', choices=CACHE_TYPE_BYTES, default='f16')
    parser.add_argument('--cache-type-v', choices=CACHE_TYPE_BYTES, default='f16')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--greedy', action='store_true')
    args = parser.parse_args()
    if args.sampling_threads is None:
        args.sampling_threads = 4 if args.cpu_sampling else 1
    source = args.input.resolve()
    destination = (args.output or source.with_name(source.stem + '.translated.jsonl')).resolve()
    summary = destination.with_suffix('.summary.json')
    log_path = destination.with_suffix('.log')
    config_path = destination.with_suffix('.config.json')
    check_distinct_paths({'input': source, 'output': destination, 'summary': summary,
                          'log': log_path, 'config': config_path})
    with source.open(encoding='utf-8-sig') as stream:
        count = sum(bool(line.strip()) for line in stream)
    if count == 0:
        parser.error('input contains no requests')
    config = resolve_config(ROOT, mode='batch', profile=args.profile,
                            parallel=min(args.parallel, count) if args.parallel is not None else None,
                            context=args.context, ubatch=args.ubatch, gpu=args.gpu, binary_dir=args.binary_dir,
                            model_override=args.model, cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v)
    parallel = min(config['parallel'], count)
    if args.max_tokens < 1 or args.context <= args.max_tokens:
        parser.error('context must exceed positive max-tokens')
    check_distinct_paths({'model': Path(config['model']), 'input': source, 'output': destination,
                          'summary': summary, 'log': log_path, 'config': config_path})
    config.update(actual_parallel=parallel, sampling_threads=args.sampling_threads,
                  gpu_prefix=not args.cpu_sampling)
    env = os.environ.copy()
    env['PATH'] = str(ROOT/'runtime/cuda') + os.pathsep + str(ROOT/'runtime/msvc') + os.pathsep + env['PATH']
    env.update(CUDA_CACHE_PATH=str(ROOT/'cache/cuda'), LLAMA_HYMT_FUSED_PROJ='1',
               LLAMA_HYMT_SPARSE_PENALTIES='1', LLAMA_HYMT_SAMPLING_SNAPSHOT='1', LLAMA_HYMT_BATCH_SNAPSHOT='1',
               GGML_CUDA_HYMT_DISABLE_ROPE_NORM='0', GGML_CUDA_HYMT_ROPE_NORM_STRICT='1',
               GGML_CUDA_HYMT_EAGER_GRAPHS='1', GGML_CUDA_GRAPH_OPT='0',
               GGML_CUDA_HYMT_DISABLE_TOPK='0', GGML_CUDA_HYMT_DISABLE_SCALAR_GATHER='0',
               GGML_CUDA_HYMT_DISABLE_Q8_KV_FUSION='0', GGML_CUDA_Q8_KV_SUBWARP='0')
    env['CUDA_VISIBLE_DEVICES'] = config['cuda_visible_devices']
    command = [str(Path(config['binary_dir'])/'hy-batch.exe'), '-m', config['model'],
               '--input', str(source), '--output', str(destination), '--summary', str(summary),
               '--parallel', str(parallel), '--sampling-threads', str(args.sampling_threads),
               '--context', str(args.context), '--batch-size', str(config['batch']),
               '--ubatch-size', str(config['ubatch']),
               '--max-tokens', str(args.max_tokens), '--seed', str(args.seed)]
    command += cache_type_args(args.cache_type_k, args.cache_type_v)
    if args.greedy:
        command.append('--greedy')
    if not args.cpu_sampling:
        command.append('--gpu-prefix')
    ensure_cache_type_support(command[0], args.cache_type_k, args.cache_type_v, env=env, cwd=ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    (ROOT/'cache/cuda').mkdir(parents=True, exist_ok=True)
    config.update(command=command)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Translating {count} requests; profile={config['profile']}, parallel={parallel}, "
          f"KV={args.cache_type_k}/{args.cache_type_v}", flush=True)
    for warning in config['warnings']:
        print(warning, file=sys.stderr)
    with log_path.open('wb') as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    if summary.exists() and completed.returncode in (0, 2):
        data = json.loads(summary.read_text(encoding='utf-8-sig'))
        print(f"{data['requests']} translations, {data['wall_s']:.3f} s, "
              f"{data['completion_tokens_per_second']:.1f} completion tokens/s")
        print(f"Truncated: {data['truncated']}; empty: {data['empty_outputs']}")
        print(f'Output: {destination}\nSummary: {summary}')
    else:
        print(log_path.read_text(encoding='utf-8', errors='replace')[-4000:], file=sys.stderr)
    return completed.returncode


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)
