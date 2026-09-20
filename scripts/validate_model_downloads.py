"""Validate public model downloads through the desktop worker in a fresh package.

Run after packaging has finished. Both raw checkpoints are fetched anonymously; local
model discovery is restricted to the supplied package so this cannot silently
reuse the maintainer's source weights. No GPU inference is performed here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    if (root/'payload').is_dir():
        root /= 'payload'
    sys.path.insert(0, str(root/'scripts'))
    from desktop_bridge import Bridge
    from setup_model import checkpoint_manifest, verify_checkpoint, sha256
    report_path = args.report.resolve()
    if report_path.exists():
        raise ValueError('Choose a fresh report path')
    data = report_path.with_suffix('')
    data.mkdir(parents=True, exist_ok=False)
    report = {'ok':False, 'root':str(root), 'models':{}}
    try:
        for profile in ('fast', 'compat'):
            bridge = Bridge({'root':str(root), 'data':str(data),
                'output':str(data/'progress.json'), 'model_dir':str(root/'models'),
                'model_search_roots':[], 'args':{'profile':profile}})
            item = bridge.manifest['files'][profile]
            bundle = bridge.model_path(item)
            spec = checkpoint_manifest(bridge.manifest, profile)
            target = bundle.parent/spec['checkpoint_dir']
            if target.exists() or bundle.exists() or (root/'models'/spec['checkpoint_dir']).exists():
                raise ValueError('Fresh download required; package already has '+profile+' weights')
            started = time.monotonic()
            result = bridge.download()
            if result.get('result') != 'downloaded_verified':
                raise RuntimeError('Expected a real network download, got '+str(result))
            verify_checkpoint(target,spec)
            assert not bundle.exists(), 'Raw downloads must not create a ZIP'
            # Alias the verified raw files for the separate CLI smoke test.
            checkpoint = root/'models'/spec['checkpoint_dir']
            checkpoint.mkdir()
            for name in spec['checkpoint_files']:
                os.link(target/name,checkpoint/name)
            inventory = bridge.inventory()
            selected = next(m for m in inventory['models'] if m['profile']==profile)
            assert selected['downloadable'] and selected['installed']
            report['models'][profile] = {'result':result['result'], 'wall_s':time.monotonic()-started,
                'size_bytes':spec['checkpoint_size_bytes'], 'files':{name:sha256(target/name) for name in spec['checkpoint_files']}, 'zip_created':False,
                'checkpoint_verified':True, 'desktop_inventory_installed':True}
            print(profile+' public download and checkpoint verified', flush=True)
        report.update(ok=True, repositories=bridge.manifest['repositories'])
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
