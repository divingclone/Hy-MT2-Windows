"""Build a verified NVFP4 model bundle and its schema-2 deployment manifest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
from setup_model import sha256, load_manifest

FILES = ('config.json', 'generation_config.json', 'tokenizer_config.json',
         'tokenizer.json', 'chat_template.jinja', 'model.safetensors')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New output directory')
    parser.add_argument('--append-manifest', type=Path, help='Add an INT4 compatibility checkpoint to an existing manifest')
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    quant = config.get('quantization_config', {})
    int4=quant.get('format')=='pack-quantized'
    if int4 and not args.append_manifest:
        raise ValueError('INT4 requires --append-manifest so it is registered as compat, not the NVFP4 fast profile')
    if quant.get('format') not in ('nvfp4-pack-quantized','pack-quantized') or quant.get('quant_method') != 'compressed-tensors':
        raise ValueError('Expected a calibrated compressed-tensors NVFP4 or INT4 checkpoint')
    if int4:
        groups=quant.get('config_groups',{}).values()
        if not groups or any(g['weights']['num_bits']!=4 or g['weights']['type']!='int' or g.get('input_activations') for g in groups):
            raise ValueError('Expected integer W4A16 quantization')
    records = {name: {'size_bytes': (source/name).stat().st_size, 'sha256': sha256(source/name)} for name in FILES}
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    name = 'Hy-MT2-1.8B-INT4-vllm' if int4 else 'Hy-MT2-1.8B-NVFP4-vllm'
    bundle = destination/(name+'.zip')
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for filename in FILES:
            archive.write(source/filename, filename)
    manifest = {'schema_version': 2, 'backend': 'vllm', 'repo_id': None,
        'revision': 'main', 'base_model': 'tencent/Hy-MT2-1.8B', 'license': 'apache-2.0',
        'checkpoint_dir': name, 'checkpoint_size_bytes': sum(r['size_bytes'] for r in records.values()),
        'checkpoint_files': records, 'files': {'fast': {'filename': bundle.name,
            'size_bytes': bundle.stat().st_size, 'sha256': sha256(bundle),
            'quantization': 'INT4 W4A16 GPTQ' if int4 else 'NVFP4', 'layout': 'compressed-tensors'}}}
    if args.append_manifest:
        if not int4: raise ValueError('--append-manifest is for the INT4 variant')
        original=load_manifest(args.append_manifest)
        if 'compat' in original['files']: raise ValueError('Manifest already has a compatibility checkpoint')
        original['files']['compat']=manifest['files']['fast']
        original['checkpoints']={'compat':{k:manifest[k] for k in ('checkpoint_dir','checkpoint_size_bytes','checkpoint_files')}}
        manifest=original
    path = destination/'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    load_manifest(path)
    print(bundle)


if __name__ == '__main__':
    main()
