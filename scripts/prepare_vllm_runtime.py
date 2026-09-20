"""Create a relocatable native Windows Python + vLLM runtime (no WSL).

Use --from-environment to reuse a verified venv; otherwise uv installs the pinned
dependencies. The destination must be fresh. Build FlashInfer AOT separately
on the maintainer host; end users need only the NVIDIA display driver.
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-environment',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'runtime/vllm')
    p.add_argument('--aot-source',type=Path,help='Directory containing sampling.dll and sampling.json from build_flashinfer_aot.py')
    p.add_argument('--msvc-redist',type=Path,help='Licensed x64 Microsoft.VC143.CRT redistribution directory')
    args=p.parse_args()
    target=args.output.resolve()
    if target.exists(): raise FileExistsError(f'Refusing to overwrite {target}')
    if args.msvc_redist is None or not (args.msvc_redist/'msvcp140.dll').is_file():
        raise ValueError('--msvc-redist must point to the Visual Studio x64 CRT redistributable directory (not the compiler).')
    source=args.from_environment
    if source is None:
        uv=shutil.which('uv')
        if not uv: raise ValueError('Install uv first, or provide --from-environment with the tested Windows vLLM venv.')
        source=ROOT/'.local/vllm-build'
        if source.exists(): raise FileExistsError(source)
        subprocess.run([uv,'venv','--python','3.12',str(source)],check=True)
        subprocess.run([uv,'pip','install','--python',str(source/'Scripts/python.exe'),
            '--extra-index-url','https://download.pytorch.org/whl/cu130','--index-strategy','unsafe-best-match',
            '-r',str(ROOT/'requirements-vllm-windows.txt')],check=True)
    source=source.resolve()
    info=json.loads(subprocess.check_output([str(source/'Scripts/python.exe'),'-c',
        'import sys,json,importlib.metadata as m; print(json.dumps({"base":sys.base_prefix,"vllm":m.version("vllm"),"torch":m.version("torch")}))'],text=True))
    if info['vllm']!='0.29.0+cu132' or info['torch']!='2.11.0+cu130':
        raise ValueError(f'Untested runtime versions: {info}')
    base=Path(info['base'])
    # Real copies avoid hardlink changes propagating into the development venv.
    shutil.copytree(base,target,ignore=shutil.ignore_patterns('__pycache__','.git','site-packages','Scripts','test','tests'))
    packages=target/'Lib/site-packages'
    shutil.copytree(source/'Lib/site-packages',packages,
        ignore=shutil.ignore_patterns('__pycache__','__editable__*','direct_url.json'))
    (target/'Scripts').mkdir(exist_ok=True)
    shutil.copyfile(source/'Scripts/ninja.exe',target/'Scripts/ninja.exe')
    for dll in args.msvc_redist.glob('*.dll'):
        shutil.copy2(dll,target/dll.name)
    shutil.copy2(packages/'torch/lib/cudart64_13.dll',packages/'triton/backends/nvidia/bin/cudart64_13.dll')
    if args.aot_source:
        import hashlib
        evidence=json.loads((args.aot_source/'sampling.json').read_text(encoding='utf-8'))
        binary=args.aot_source/'sampling.dll'
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=evidence['sha256'] or evidence['flashinfer']!='0.6.11.post3':
            raise ValueError('AOT provenance or version mismatch')
        aot=packages/'flashinfer/data/aot/sampling'
        aot.mkdir(parents=True,exist_ok=True)
        for name in ('sampling.dll','sampling.json'): shutil.copy2(args.aot_source/name,aot/name)
    for name in ('hymt_vllm_native.py','hymt_vllm_model.py'):
        shutil.copyfile(ROOT/'scripts/vllm_native_plugin'/name,packages/name)
    meta=packages/'hymt_vllm_native-0.1.0.dist-info'
    meta.mkdir(exist_ok=True)
    (meta/'METADATA').write_text('Metadata-Version: 2.1\nName: hymt-vllm-native\nVersion: 0.1.0\n',encoding='utf-8')
    (meta/'entry_points.txt').write_text('[vllm.general_plugins]\nhymt_native = hymt_vllm_native:register\n',encoding='utf-8')
    compiler=packages/'flashinfer/jit/cpp_ext.py'
    text=compiler.read_text(encoding='utf-8')
    if 'errors="replace"' not in text:
        needle='encoding="utf-8",'
        if needle not in text: raise ValueError('FlashInfer compiler-output patch no longer applies')
        text=text.replace(needle,needle+'\n            errors="replace",')
        compiler.write_text(text,encoding='utf-8')
    (target/'hymt-runtime.json').write_text(json.dumps({'backend':'vllm','vllm':info['vllm'],
        'torch':info['torch'],'python':'3.12','windows_native':True,'host_requirements':['NVIDIA display driver 596.36+'],
        'compiler':'bundled Triton TinyCC/PTXAS','flashinfer':'prebuilt sampling AOT; run build_flashinfer_aot.py if not supplied'},indent=2),encoding='utf-8')
    # Every import must now use this relocated prefix rather than the old venv.
    subprocess.run([str(target/'python.exe'),'-E','-s','-c',
        'import sys, pathlib, torch, vllm, hymt_vllm_native; assert pathlib.Path(torch.__file__).is_relative_to(sys.prefix); print(sys.prefix, torch.__version__, vllm.__version__)'],check=True)
    print(f'Ready: {target}')

if __name__=='__main__': main()
