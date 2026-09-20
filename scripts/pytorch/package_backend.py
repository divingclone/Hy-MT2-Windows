"""Package a validated custom payload using the existing desktop executable."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_desktop
from package_windows import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.payload.resolve()
    if not (payload / 'runtime/vllm/Lib/site-packages/torch/hymt-build.json').is_file():
        parser.error('A custom PyTorch payload with completed validation is required')
    output = args.output.resolve()
    if output == payload or payload in output.parents:
        parser.error('The output directory must be outside the payload')
    config = json.loads((build_desktop.DESKTOP / 'src-tauri/tauri.conf.json').read_text(encoding='utf-8'))
    build_desktop.PAYLOAD = payload
    archive = build_desktop.package_portable(output, config['version'])
    (output / 'SHA256SUMS.txt').write_text(f'{sha256(archive)}  {archive.name}\n', encoding='utf-8')
    print(f'Portable: {archive}', flush=True)


if __name__ == '__main__':
    main()
