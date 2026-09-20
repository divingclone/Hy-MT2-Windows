"""Record Windows DLL dependencies and proposed release exclusions."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dumpbin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
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
