"""Managed native Windows vLLM API. Default: NVFP4 CUTLASS and INT8 KV."""
import argparse,json,os,re,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from gpu_config import resolve_config
from vllm_runtime import environment,server_command,stop_process_tree
ROOT=Path(__file__).resolve().parents[1]
def process_creation_filetime(process):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    timestamps = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(int(process._handle), *(ctypes.byref(value) for value in timestamps)):
        raise ctypes.WinError(ctypes.get_last_error())
    created = timestamps[0]
    return str((created.dwHighDateTime << 32) | created.dwLowDateTime)

def check_managed_label(label):
    pid_file = ROOT/'results'/f'{label}.pid'
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding='ascii').strip())
        if pid <= 0:
            raise ValueError('invalid PID')
    except ValueError as error:
        raise RuntimeError(f'Invalid managed PID file: {pid_file}') from error
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # PID does not exist.
            return
        raise RuntimeError(f'Cannot verify existing PID {pid} (Windows error {error}); choose another label')
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259:
            raise RuntimeError(f'Label {label!r} already manages a live process (PID {pid}); stop it or choose another label')
    finally:
        kernel.CloseHandle(handle)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('auto','fast','quality','compat'),default='auto')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--parallel',type=int,default=32)
    parser.add_argument('--context',type=int,default=2048)
    parser.add_argument('--batch-tokens',type=int,default=2048)
    parser.add_argument('--kv-cache-dtype',choices=('int8_per_token_head','bfloat16','fp8_per_token_head'),default='int8_per_token_head')
    parser.add_argument('--kv-gib',type=float)
    parser.add_argument('--memory-percent',type=int,default=75)
    parser.add_argument('--gpu')
    parser.add_argument('--port',type=int,default=18080)
    parser.add_argument('--label',default='server')
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+',args.label) or not 1<=args.port<=65535:
        parser.error('Invalid label or port')
    check_managed_label(args.label)
    with socket.socket() as probe: probe.bind(('127.0.0.1',args.port))
    plan=resolve_config(ROOT,profile=args.profile,model_override=args.model,parallel=args.parallel,
        context=args.context,ubatch=args.batch_tokens,gpu=args.gpu,cache_type_k=args.kv_cache_dtype,
        kv_gib=args.kv_gib,memory_percent=args.memory_percent)
    env=environment(ROOT,gpu=plan['gpu']['uuid'])
    # Credentials are supplied through the environment, never persisted in config/logs.
    if os.environ.get('VLLM_API_KEY'): env['VLLM_API_KEY']=os.environ['VLLM_API_KEY']
    command=server_command(plan,root=ROOT,port=args.port)
    results=ROOT/'results'
    results.mkdir(exist_ok=True)
    cfg={**plan,'command':command,'mode':'server','label':args.label,'port':args.port,'executable':command[0]}
    config_path=results/f'{args.label}.config.json'
    if not args.background:
        config_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        proc=subprocess.Popen(command,env=env,cwd=ROOT)
        try: return proc.wait()
        finally: stop_process_tree(proc)
    with (results/f'{args.label}.stdout.log').open('wb') as out,(results/f'{args.label}.stderr.log').open('wb') as err:
        proc=subprocess.Popen(command,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    cfg.update(managed_pid=proc.pid,process_creation_filetime=process_creation_filetime(proc))
    config_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (results/f'{args.label}.pid').write_text(str(proc.pid),encoding='ascii')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline=time.monotonic()+600
    try:
        while proc.poll() is None and time.monotonic()<deadline:
            try:
                with opener.open(f'http://127.0.0.1:{args.port}/health',timeout=2) as response:
                    if response.status==200:
                        print(f'Ready: http://127.0.0.1:{args.port}; vLLM PID {proc.pid}; {plan["kernel"]}; KV={plan["cache"]}',flush=True)
                        return 0
            except OSError: pass
            time.sleep(1)
        raise RuntimeError(f'vLLM failed or timed out; see {results}/{args.label}.stderr.log')
    except BaseException:
        stop_process_tree(proc)
        (results/f'{args.label}.pid').unlink(missing_ok=True)
        raise

if __name__=='__main__':
    try: raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(1)
