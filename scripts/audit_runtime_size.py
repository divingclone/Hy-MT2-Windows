"""Record Windows DLL dependencies and proposed release exclusions."""
import argparse
from collections import defaultdict
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import zipfile

from runtime_filter import exclusion_reason


def audit_archive(archive):
    """Estimate new exclusions using the existing ZIP's actual compressed sizes."""
    groups = defaultdict(lambda: {'files': 0, 'bytes': 0, 'compressed_bytes': 0})
    removed = []
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            if item.is_dir() or not item.filename.startswith('payload/'):
                continue
            reason = exclusion_reason(item.filename.removeprefix('payload/'), include_webview2=True)
            if reason is None:
                continue
            group = groups[reason]
            group['files'] += 1
            group['bytes'] += item.file_size
            group['compressed_bytes'] += item.compress_size
            removed.append({'path': item.filename, 'reason': reason,
                            'bytes': item.file_size, 'compressed_bytes': item.compress_size})
    return {'archive': Path(archive).name, 'archive_bytes': Path(archive).stat().st_size,
            'groups': dict(groups), 'removed': removed,
            'removed_bytes': sum(item['bytes'] for item in removed),
            'removed_compressed_bytes': sum(item['compressed_bytes'] for item in removed),
            'note': 'Compressed file data only; final ZIP also changes directory/header overhead.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dumpbin', type=Path)
    source.add_argument('--archive', type=Path, help='Audit additional exclusions in a desktop ZIP')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.archive:
        report = audit_archive(args.archive)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k != 'removed'}, indent=2))
        return
    root = Path(__file__).resolve().parents[1]
    packages = root/'runtime/vllm/Lib/site-packages'
    files = list((packages/'torch/lib').glob('*.dll')) + list((packages/'vllm').rglob('*.pyd'))
    def inspect(file):
        result = subprocess.check_output([str(args.dumpbin), '/DEPENDENTS', str(file)], text=True)
        return {'path': file.relative_to(packages).as_posix(), 'bytes': file.stat().st_size,
                'dependencies': re.findall(r'^\s+([\w.\-]+\.dll)\s*$', result, re.M|re.I)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(inspect, files))
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
