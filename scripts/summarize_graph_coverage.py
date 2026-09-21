"""Publish compact graph-coverage evidence without paths, prompts or translations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def output_signature(row):
    return tuple(row.get(key) for key in ('id', 'translation', 'completion_tokens', 'prompt_tokens', 'finish_reason'))


def summarize(directory):
    matrix = directory / 'matrix.json'
    data = json.loads(matrix.read_text(encoding='utf-8'))
    if not data.get('valid') or len(data['runs']) != data['trials'] * len(data['capture_sizes']):
        raise ValueError('Incomplete or failed benchmark matrix')
    plan = data['plan']
    result = {'source_sha256': sha256(matrix), 'concurrency': data['concurrency'],
              'protocol': 'non_streaming_json', 'model': Path(plan['model']).name,
              'gpu': {key: plan['gpu'][key] for key in ('name', 'compute_capability', 'driver_version')},
              'context': plan['context'], 'batch_tokens': plan['batch'], 'kernel': plan['kernel'],
              'kv_cache_dtype': plan['cache'], 'kv_token_capacity': plan['budget']['kv_token_capacity'],
              'candidates': [], 'greedy_comparisons': []}
    if 'runtime' in data:
        result['runtime'] = data['runtime']
    greedy_reference = None
    for candidate, capture_size in enumerate(data['capture_sizes'], 1):
        candidate_rows = [row for row in data['runs'] if row['candidate'] == candidate]
        records = []
        for row in candidate_rows:
            folder = directory / row['name']
            report = row['report']
            metrics = []
            for run in report['runs']:
                path = folder / 'bench' / (run['phase'] + '.jsonl')
                if sha256(path) != run['output_sha256']:
                    raise ValueError(f'Output hash mismatch: {path}')
                outputs = read_rows(path)
                if (len(outputs) != 512 or len({x['id'] for x in outputs}) != 512
                        or any(not x['ok'] or x['truncated'] or x['finish_reason'] != 'stop' for x in outputs)
                        or sum(x['completion_tokens'] for x in outputs) != run['metrics']['completion_tokens']
                        or not run['metrics']['valid']):
                    raise ValueError(f'Invalid output pass: {path}')
                metrics.append({'phase': run['phase'], 'metrics': run['metrics'],
                                'output_sha256': run['output_sha256']})
            log = (folder / 'stdout.log').read_text(encoding='utf-8', errors='replace')
            graph_memory = re.findall(r'Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB', log)
            records.append({'trial': row['trial'], 'initialization_s': row['initialization_s'],
                            'graph_capture': {'seconds_rounded': float(graph_memory[-1][0]),
                                              'gib_rounded': float(graph_memory[-1][1])} if graph_memory else None,
                            'warm_median': report['warm_median'], 'runs': metrics})
            if 'greedy_report' in row:
                for run in row['greedy_report']['runs']:
                    path = folder / 'greedy' / (run['phase'] + '.jsonl')
                    if sha256(path) != run['output_sha256'] or not run['metrics']['valid']:
                        raise ValueError(f'Invalid greedy pass: {path}')
                    signatures = [output_signature(x) for x in read_rows(path)]
                    if greedy_reference is None:
                        greedy_reference = signatures
                    if len(signatures) != len(greedy_reference):
                        raise ValueError('Greedy request counts differ')
                    result['greedy_comparisons'].append({'candidate': candidate, 'trial': row['trial'],
                        'phase': run['phase'], 'requests': len(signatures),
                        'exact_matches_to_first_baseline': sum(x == y for x, y in zip(signatures, greedy_reference)),
                        'output_sha256': run['output_sha256']})
        tps = [row['warm_median']['completion_tokens_per_second'] for row in records]
        p95 = [statistics.median(run['metrics']['latency_s']['p95'] for run in row['runs'][1:]) for row in records]
        result['candidates'].append({'candidate': candidate, 'capture_size': capture_size,
            'tps_median': statistics.median(tps), 'tps_process_range': [min(tps), max(tps)],
            'latency_p95_s_median': statistics.median(p95),
            'initialization_s_median': statistics.median(row['initialization_s'] for row in records),
            'records': records})
    result['dataset_sha256'] = data['runs'][0]['report']['dataset_sha256']
    greedy = data['runs'][0]['report']['arguments']['greedy']
    result['sampling'] = {'temperature': 0 if greedy else .7,
                          'top_p': 1 if greedy else .6, 'top_k': 1 if greedy else 20,
                          'max_tokens': 512, 'repetition_penalty': 1.05, 'seed': '42 + request index'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: row[key] for key in ('capture_size', 'tps_median', 'latency_p95_s_median')}
                      for row in result['candidates']], indent=2))


if __name__ == '__main__':
    main()
