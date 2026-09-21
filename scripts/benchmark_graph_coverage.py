"""Compare CUDA Graph coverage at fixed API concurrency and KV capacity.

Each candidate gets an independent server, a full warmup pass, and repeated
512-request passes. Trial order alternates. No GPU profiler or polling process
runs during timing. Raw JSONL responses and failed candidates are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

from gpu_config import resolve_config
from vllm_runtime import ROOT, environment, server_command, stop_process_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(32, 256), default=32)
    parser.add_argument('--capture-sizes', type=int, nargs='+', default=[64, 256],
                        help='Explicit maximum captured token counts; compare equal values for a control')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--port', type=int, default=18086)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--check-greedy', action='store_true',
                        help='After timing, retain two additional greedy passes for output comparisons')
    args = parser.parse_args()
    if (args.trials < 1 or args.repeats < 1 or not 1 <= args.port <= 65535
            or any(size < args.concurrency or size > 2048 for size in args.capture_sizes)):
        parser.error('Positive trials/repeats and capture sizes between concurrency and 2048 required')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Resolve once so every candidate has identical model, context and KV bytes.
    plan = resolve_config(ROOT, profile='fast', parallel=args.concurrency,
                          context=2048, ubatch=2048)
    env = environment(ROOT, gpu=plan['gpu']['uuid'])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f'http://127.0.0.1:{args.port}'
    report = {'plan': plan, 'concurrency': args.concurrency, 'repeats': args.repeats,
              'trials': args.trials, 'capture_sizes': args.capture_sizes,
              'notes': ['Identical KV capacity and maximum active sequences across candidates.',
                        'No GPU telemetry or profiling during timed passes.',
                        'One full warmup pass followed by measured passes per independent server.',
                        'Each trial reverses candidate order; no cached translations.'],
              'runs': []}
    torch_root = ROOT / 'runtime/vllm/Lib/site-packages/torch'
    with (torch_root / 'lib/torch_cuda.dll').open('rb') as stream:
        dll_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    marker_path = torch_root / 'hymt-build.json'
    report['runtime'] = {'torch_cuda_sha256': dll_sha,
                         'custom_build_marker_present': marker_path.is_file()}
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        report['runtime']['profile'] = marker['profile']
        report['runtime']['inference_validated'] = marker['inference_validated']

    def save():
        (output / 'matrix.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    save()
    for trial in range(1, args.trials + 1):
        candidates = list(enumerate(args.capture_sizes))
        if trial % 2 == 0:
            candidates.reverse()
        for index, size in candidates:
            name = f'candidate{index + 1}-graph{size}-t{trial}'
            directory = output / name
            directory.mkdir()
            command = server_command(plan, port=args.port)
            # Override the one config object, avoiding duplicate CLI options.
            if '--compilation-config' in command:
                pos = command.index('--compilation-config') + 1
                compilation = json.loads(command[pos])
                compilation['max_cudagraph_capture_size'] = size
                command[pos] = json.dumps(compilation)
            else:
                command += ['--compilation-config', json.dumps({'max_cudagraph_capture_size': size})]
            record = {'name': name, 'candidate': index + 1, 'capture_size': size,
                      'trial': trial, 'command': command}
            report['runs'].append(record)
            save()
            with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
                started = time.perf_counter()
                process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=out, stderr=err,
                                           stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'Server exited: {process.returncode}')
                        try:
                            with opener.open(base_url + '/health', timeout=2) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        if time.perf_counter() - started > 600:
                            raise TimeoutError('Server startup exceeded 600 seconds')
                        time.sleep(.5)
                    record['initialization_s'] = time.perf_counter() - started
                    bench_command = [sys.executable, '-X', 'utf8', str(ROOT / 'scripts/benchmark_backend.py'),
                                     '--backend', 'vllm', '--url', base_url, '--concurrency', str(args.concurrency),
                                     '--client', 'aiohttp', '--repeats', str(args.repeats),
                                     '--output', str(directory / 'bench')]
                    if args.greedy:
                        bench_command.append('--greedy')
                    with (directory / 'bench.log').open('wb') as log:
                        subprocess.run(bench_command, stdout=log, stderr=log, cwd=ROOT, check=True, timeout=900)
                    record['report'] = json.loads((directory / 'bench/report.json').read_text(encoding='utf-8'))
                    print(name, record['report']['warm_median'], flush=True)
                    if args.check_greedy:
                        greedy_command = bench_command.copy()
                        greedy_command[greedy_command.index('--output') + 1] = str(directory / 'greedy')
                        greedy_command[greedy_command.index('--repeats') + 1] = '1'
                        if '--greedy' not in greedy_command:
                            greedy_command.append('--greedy')
                        with (directory / 'greedy.log').open('wb') as log:
                            subprocess.run(greedy_command, stdout=log, stderr=log, cwd=ROOT, check=True, timeout=900)
                        record['greedy_report'] = json.loads((directory / 'greedy/report.json').read_text(encoding='utf-8'))
                except Exception as error:
                    record['error'] = repr(error)
                    print(name, record['error'], flush=True)
                finally:
                    stop_process_tree(process)
                    save()
    report['valid'] = all('error' not in row for row in report['runs'])
    if report['valid']:
        report['summary'] = []
        for index, size in enumerate(args.capture_sizes, 1):
            rows = [row for row in report['runs'] if row['candidate'] == index]
            tps = [row['report']['warm_median']['completion_tokens_per_second'] for row in rows]
            report['summary'].append({'candidate': index, 'capture_size': size,
                                      'tps_median': statistics.median(tps), 'tps_range': [min(tps), max(tps)],
                                      'initialization_s_median': statistics.median(row['initialization_s'] for row in rows)})
    save()
    if not report['valid']:
        raise RuntimeError(f'Invalid candidates retained in {output}/matrix.json')


if __name__ == '__main__':
    main()
