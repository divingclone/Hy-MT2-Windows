"""Prepare pinned public OPUS-100 en/zh calibration and held-out mixed requests.

Requires pyarrow (development only). Raw corpus and derived text stay in the
selected output directory; the repository should contain only hashes/metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import urllib.request

from translate import translation_prompt

REVISION = '805090dc28bf78897da9641cdf08b61287580df9'
DATASET = 'https://huggingface.co/datasets/Helsinki-NLP/opus-100'
HASHES = {
    'validation': 'ce0428a37c741a3c55ae707ca7b12fc0410b015212ad604f9b305f810188434c',
    'test': '571b811b20859d801561f62a21cc0e8a2fb034dd54bc960bb4b879dc6345fd19',
}
BOS = '<｜hy_begin▁of▁sentence｜>'
USER = '<｜hy_User｜>'
ASSISTANT = '<｜hy_Assistant｜>'
EOT = '<｜hy_place▁holder▁no▁2｜>'
# Literal model control strings in corpus content would be parsed as controls
# by llama-imatrix --parse-special. Exclude their aligned pair, never rewrite
# just one side. These patterns cover Hy-MT2 control and user-defined tokens.
CONTROL_TOKEN = re.compile(r'<｜hy_[^>]*>|</?(?:think|answer|tool_calls?|tool_responses?|tool_sep)>|/no_think|/think')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def deduplicate(rows: list, forbidden: set | None = None) -> list:
    seen = set(forbidden or ())
    result = []
    for index, item in enumerate(rows):
        pair = item.get('translation') if isinstance(item, dict) else None
        if not isinstance(pair, dict):
            continue
        if not all(isinstance(pair.get(lang), str) and pair[lang].strip() for lang in ('en', 'zh')):
            continue
        if any(CONTROL_TOKEN.search(pair[lang]) for lang in ('en', 'zh')):
            continue
        keys = {(lang, normalized(pair[lang])) for lang in ('en', 'zh')}
        if keys & seen:
            continue
        seen.update(keys)
        result.append({'index': index, 'en': pair['en'], 'zh': pair['zh']})
    return result


def requests(rows: list, split: str, count: int, seed: int) -> list:
    """Sample without replacement; vary sentence count, then shuffle requests."""
    if count < 1:
        raise ValueError('Request count must be positive')
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    result, direction_counts = [], Counter()
    for i in range(count):
        # Half single sentences; other requests contain different aligned pairs.
        size = 1 if i % 2 == 0 else rng.randint(2, 6) if i % 4 == 1 else rng.randint(7, 16)
        if len(pool) < size:
            raise ValueError('Requested more distinct sentence pairs than this split contains')
        chosen, pool = pool[:size], pool[size:]
        bucket = 'single' if size == 1 else 'medium' if size <= 6 else 'long'
        # Balance each length class independently. An index-modulo direction
        # rule would correlate medium with en->zh and long with zh->en.
        source, target = ('en', 'zh') if direction_counts[bucket] % 2 == 0 else ('zh', 'en')
        direction_counts[bucket] += 1
        result.append({
            'id': f'opus-{split}-{i:04d}',
            'text': '\n'.join(x[source].strip() for x in chosen),
            'target_lang': 'Chinese' if target == 'zh' else 'English',
            'reference': '\n'.join(x[target].strip() for x in chosen),
            'source_lang': source, 'sentence_indices': [x['index'] for x in chosen],
            'sentence_count': size, 'bucket': bucket,
            'split': split,
        })
    rng.shuffle(result)
    return result


def calibration_text(rows: list) -> str:
    """Match a native generation prompt, then the reference and terminal EOG.

    The non-generation chat-template suffix (placeholder 8) is deliberately
    absent: these are teacher-forced continuations of actual inference prompts.
    """
    for row in rows:
        if any(CONTROL_TOKEN.search(row[field]) for field in ('text', 'reference')):
            raise ValueError('Corpus text contains a model control token')
    return '\n'.join(BOS + USER + translation_prompt(x['text'], x['target_lang']) + ASSISTANT + x['reference'] + EOT
                     for x in rows)


def bucket_languages(rows: list) -> dict:
    return {bucket: dict(Counter(row['source_lang'] for row in rows if row['bucket'] == bucket))
            for bucket in ('single', 'medium', 'long')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--calibration-count', type=int, default=256)
    parser.add_argument('--evaluation-count', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260918)
    args = parser.parse_args()
    if min(args.calibration_count, args.evaluation_count) < 1:
        parser.error('Counts must be positive')
    out = args.output_dir.resolve()
    names = ('calibration.jsonl', 'evaluation.jsonl', 'calibration.txt', 'dataset.json')
    if any((out / name).exists() for name in names):
        parser.error('Output files already exist; choose a new directory')
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    data, sources = {}, {}
    for split, expected in HASHES.items():
        url = f'{DATASET}/resolve/{REVISION}/en-zh/{split}-00000-of-00001.parquet?download=true'
        path = raw / f'{split}.parquet'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f'Download hash mismatch: {split}')
            path.write_bytes(payload)
        if digest(path) != expected:
            raise ValueError(f'Existing raw file hash mismatch: {path}')
        data[split] = pq.read_table(path).to_pylist()
        sources[split] = {'url': url, 'sha256': expected, 'rows': len(data[split])}
    # Keep evaluation independent even if the upstream splits contain duplicates.
    evaluation_pool = deduplicate(data['test'])
    forbidden = {(lang, normalized(pair[lang])) for pair in evaluation_pool for lang in ('en', 'zh')}
    calibration_pool = deduplicate(data['validation'], forbidden)
    calibration = requests(calibration_pool, 'validation', args.calibration_count, args.seed - 1)
    evaluation = requests(evaluation_pool, 'test', args.evaluation_count, args.seed)
    for name, values in [('calibration', calibration), ('evaluation', evaluation)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in values), encoding='utf-8')
    text = calibration_text(calibration)
    (out / 'calibration.txt').write_text(text, encoding='utf-8')
    manifest = {
        'schema_version': 2, 'dataset': DATASET, 'revision': REVISION, 'sources': sources, 'seed': args.seed,
        'calibration_seed': args.seed - 1, 'evaluation_seed': args.seed,
        'calibration_split': 'validation', 'evaluation_split': 'test',
        'deduplicated_pool_rows': {'validation': len(calibration_pool), 'test': len(evaluation_pool)},
        'calibration_requests': len(calibration), 'evaluation_requests': len(evaluation),
        'calibration_buckets': dict(Counter(x['bucket'] for x in calibration)),
        'evaluation_buckets': dict(Counter(x['bucket'] for x in evaluation)),
        'calibration_bucket_source_languages': bucket_languages(calibration),
        'evaluation_bucket_source_languages': bucket_languages(evaluation),
        'length_policy': {'unit': 'aligned sentence pairs, not tokens', 'single': [1, 1],
                          'medium': [2, 6], 'long': [7, 16], 'request_proportions': [0.5, 0.25, 0.25],
                          'source_language_balance': 'Each bucket independently alternates en/zh; difference at most one request'},
        'files': {name: digest(out / name) for name in names[:-1]},
        'notes': [
            'Use calibration.txt with llama-imatrix --parse-special and the original high precision model.',
            'Calibration uses the native inference prompt prefix plus reference and terminal EOG; literal model control strings in corpus content are excluded.',
            'Sentence-count buckets do not define token budgets. Measure actual token lengths and retain the same context/output limits across candidates.',
            'References come from public OPUS parallel data, whose alignment/reference quality can vary.',
            'Mixed requests are constructed from distinct aligned sentences, not coherent source documents or production traces.',
            'Do not label scores on these mixed subsets as full official OPUS-100 benchmark scores.',
            'Cross-split normalized en or zh duplicates excluded from calibration; this does not establish absence from model pretraining.',
            'OPUS-100 aggregates corpora with varying terms; upstream dataset card lists license as unknown. Raw/derived texts are local evaluation artifacts.',
        ],
    }
    (out / 'dataset.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
