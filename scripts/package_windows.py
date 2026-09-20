"""Package native Windows vLLM, excluding old engines, caches and credentials."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,zipfile
from dataclasses import dataclass
from pathlib import Path
from runtime_filter import excluded
from runtime_patches import VISION_ATTENTION, patch_release_runtime
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=('serve.py','serve_vllm.py','serve.ps1','stop-server.ps1','run-python.cmd','gpu_config.py',
    'vllm_runtime.py','http_transport.py','translate.py','translate_batch.py','vllm_batch_worker.py','setup_model.py',
    'prepare_vllm_runtime.py','desktop_bridge.py','desktop_update.py','prepare_webview_runtime.py','webview_progress.py')
LAUNCHERS=('start-server.cmd','stop-server.cmd','translate-batch.cmd','setup-model.cmd','gpu-info.cmd','setup-runtime.cmd')

@dataclass(frozen=True)
class Entry:
    source:Path
    relative:str
    component:str

def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def runtime_problems(root):
    """Guard exclusions even when the caller reuses an older staging tree."""
    try:
        runtime=json.loads((Path(root)/'runtime/vllm/hymt-runtime.json').read_text(encoding='utf-8'))
        if not isinstance(runtime,dict):
            return ['Missing or invalid vLLM runtime provenance']
        if (runtime.get('vllm'),runtime.get('torch'))!=('0.29.0+cu132','2.11.0+cu130'):
            return ['Runtime exclusions require revalidation for these vLLM/PyTorch versions']
    except (OSError,ValueError):
        return ['Missing or invalid vLLM runtime provenance']
    torch_root=Path(root)/'runtime/vllm/Lib/site-packages/torch'
    custom_path=torch_root/'hymt-build.json'
    if custom_path.exists() or runtime.get('torch_custom_build'):
        try:
            custom=json.loads(custom_path.read_text(encoding='utf-8'))
            if (custom.get('profile')!='hymt-cuda-delay-v1' or
                    custom.get('binary_compatibility_checked') is not True or
                    custom.get('inference_validated') is not True):
                return ['Custom PyTorch requires completed binary and inference validation']
            if sha256(torch_root/'lib/torch_cuda.dll')!=custom['custom_dll_sha256']:
                return ['Custom PyTorch CUDA DLL differs from the validated build']
            if sha256(torch_root/'cuda/__init__.py')!=custom['cuda_python_sha256']:
                return ['Custom PyTorch architecture metadata differs from the validated build']
            for name,expected in custom['build_provenance']['base_libraries'].items():
                if sha256(torch_root/'lib'/name)!=expected:
                    return ['Custom PyTorch requires the original matching CPU/c10/Python libraries']
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            return ['Missing or invalid custom PyTorch provenance']
    return []


def make_plan(root,binary_dir=None,python_dir=None,include_models=False,include_webview2=False):
    root=Path(root).resolve(); entries=[]; problems=runtime_problems(root)
    def add(source,destination,component):
        if excluded(destination,include_webview2=include_webview2):return
        source=Path(source)
        if not source.is_file():problems.append(f'Missing {source}');return
        if source.is_symlink():raise ValueError(f'Linked package source: {source}')
        entries.append(Entry(source,destination,component))
    for name in LAUNCHERS:add(root/name,name,'launcher')
    for name in SCRIPTS:add(root/'scripts'/name,'scripts/'+name,'application')
    for file in sorted((root/'scripts/vllm_native_plugin').glob('*.py')):
        add(file,'scripts/vllm_native_plugin/'+file.name,'runtime setup')
    for name in ('README.md','LICENSE','requirements-vllm-windows.txt'):add(root/name,name,'documentation')
    for folder in ('docs','licenses','examples','benchmarks'):
        for file in sorted((root/folder).glob('*')):
            if file.suffix in ('.md','.txt','.jsonl','.json') and file.is_file() and 'output' not in file.name:
                add(file,file.relative_to(root).as_posix(),'documentation')
    add(root/'models/manifest.json','models/manifest.json','model manifest')
    if include_models:
        manifest=json.loads((root/'models/manifest.json').read_text(encoding='utf-8'))
        for item in manifest['files'].values():
            file=root/'models'/item['filename'];add(file,'models/'+item['filename'],'model bundle')
            if file.exists() and sha256(file)!=item['sha256']:problems.append('Model bundle digest differs from manifest')
    python_dir=Path(python_dir) if python_dir else root/'runtime/python'
    if not (python_dir/'python.exe').is_file():python_dir/='cpython-3.12-windows-x86_64-none'
    for source,destination in ((python_dir,'runtime/python'),(root/'runtime/vllm','runtime/vllm')):
        if not (source/'python.exe').is_file():problems.append(f'Missing runtime: {source}');continue
        for file in sorted(source.rglob('*')):
            if not file.is_file():continue
            parts=file.relative_to(source).parts
            # HIP-only generated source is unused by our NVIDIA runtime and
            # contains paths too long for Windows Explorer's ZIP extractor.
            if parts[:7]==('Lib','site-packages','tilelang','3rdparty','composable_kernel','library','src'):continue
            if any(p in ('__pycache__','.git') for p in parts) or file.name=='direct_url.json':continue
            if file.suffix in ('.pyc','.pyo','.log') or file.name.startswith('__editable__'):continue
            if parts[0]=='Scripts' and file.name!='ninja.exe':continue
            add(file,destination+'/'+file.relative_to(source).as_posix(),'runtime')
    for name in ('triton/runtime/tcc/tcc.exe','triton/backends/nvidia/bin/cudart64_13.dll','flashinfer/data/aot/sampling/sampling.dll'):
        if not (root/'runtime/vllm/Lib/site-packages'/name).is_file(): problems.append('Missing portable inference dependency: '+name)
    if include_webview2:
        browser=root/'runtime/webview2'
        if not (browser/'msedgewebview2.exe').is_file(): problems.append('Missing fixed WebView2 runtime')
        for file in sorted(browser.rglob('*')):
            if file.is_file(): add(file,'runtime/webview2/'+file.relative_to(browser).as_posix(),'WebView2')
    return entries,problems,{'backend':'vllm','include_models':include_models,'include_webview2':include_webview2}

def verify_python(root):
    for folder in ('python','vllm'):
        subprocess.run([str(Path(root)/'runtime'/folder/'python.exe'),'-E','-s','-X','utf8','-c',
            'import pathlib,sys,json,ctypes,ssl; assert pathlib.Path(sys.executable).parent==pathlib.Path(sys.prefix); print(sys.prefix)'],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/vllm-runtime')
    model_options=parser.add_mutually_exclusive_group()
    model_options.add_argument('--include-models',action='store_true',help='Explicitly build an offline package with both model ZIPs')
    model_options.add_argument('--no-models',action='store_true',help='Compatibility option; model-free is already the default')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--include-webview2',action='store_true',help='Include app-local WebView2 for offline first launch')
    parser.add_argument('--no-zip',action='store_true');args=parser.parse_args()
    entries,problems,_=make_plan(ROOT,include_models=args.include_models,include_webview2=args.include_webview2)
    if problems:raise ValueError('\n'.join(problems))
    print(f'{len(entries)} files, {sum(e.source.stat().st_size for e in entries)/1024**3:.2f} GiB unpacked')
    if args.dry_run:return
    target=args.output.resolve();target.mkdir(parents=True,exist_ok=False);records=[]
    for entry in entries:
        path=target/entry.relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(entry.source,path)
        records.append({'path':entry.relative,'bytes':path.stat().st_size,'sha256':sha256(path)})
    patch_release_runtime(target)
    # The release import patch changes one staged Python file after copying.
    for record in records:
        if record['path']==VISION_ATTENTION:
            path=target/record['path']
            record.update(bytes=path.stat().st_size,sha256=sha256(path))
    verify_python(target)
    (target/'manifest.sha256.json').write_text(json.dumps({'backend':'vllm','files':records},indent=2),encoding='utf-8')
    if not args.no_zip:
        with zipfile.ZipFile(target.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
            for file in target.rglob('*'):
                if file.is_file():archive.write(file,file.relative_to(target.parent).as_posix())
    print(target)

if __name__=='__main__':main()
