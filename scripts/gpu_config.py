"""Read-only NVIDIA probe and token-pool budgeting for native Windows vLLM."""
from __future__ import annotations
import argparse, csv, io, json, math, os, re, shutil, subprocess
from pathlib import Path
MIB=1024**2
INT32_MAX=2**31-1
KV_ELEMENTS_PER_TOKEN=32*4*128
CACHE_TYPE_BYTES={'bfloat16':2.0,'int8_per_token_head':132/128,'fp8_per_token_head':132/128}
PARALLEL_STEPS=(1,2,4,8,16,32,64,128,256)
class ConfigError(ValueError):
    """An actionable configuration or device error, suitable for CLI display."""

def _positive_int(value: object, label: str, maximum: int = INT32_MAX) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ConfigError(f"{label}必须是1至{maximum}的整数。")
    return value

def _capability(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d)", str(value).strip())
    if not match:
        raise ConfigError(f"无法识别GPU计算能力 {value!r}；请更新NVIDIA驱动后重新运行nvidia-smi。")
    return int(match[1]), int(match[2])

def _version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)*", str(value)):
        raise ConfigError(f"无法识别驱动版本 {value!r}。")
    return tuple(int(part) for part in value.split("."))

def _version_at_least(actual: str, minimum: str) -> bool:
    first, second = _version(actual), _version(minimum)
    width = max(len(first), len(second))
    return first + (0,) * (width - len(first)) >= second + (0,) * (width - len(second))

def parse_smi_csv(output: str) -> list[dict]:
    """Pure parser for the exact seven-column nvidia-smi query below."""
    devices = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row or all(not part.strip() for part in row):
            continue
        if len(row) != 7:
            raise ConfigError("nvidia-smi返回了无法识别的GPU信息；请检查驱动并重新运行nvidia-smi -L。")
        index, uuid, name, capability, total, free, driver = (part.strip() for part in row)
        try:
            index_number = int(index)
            total_number, free_number = float(total), float(free)
        except ValueError as error:
            raise ConfigError("nvidia-smi未提供有效显存信息；请关闭冲突的GPU任务并检查驱动。") from error
        if (index_number < 0 or not uuid.startswith("GPU-") or not name or
                not math.isfinite(total_number) or not math.isfinite(free_number) or
                not 0 <= free_number <= total_number or total_number <= 0):
            raise ConfigError("nvidia-smi的设备或显存数据无效；请检查驱动并重新运行nvidia-smi。")
        _capability(capability)
        _version(driver)
        devices.append({"index": index_number, "uuid": uuid, "name": name,
                        "compute_capability": capability, "total_memory_mib": total_number,
                        "free_memory_mib": free_number, "driver_version": driver})
    if not devices:
        raise ConfigError("未检测到NVIDIA显卡。请确认已安装受支持的NVIDIA显卡和驱动，并运行nvidia-smi -L检查。")
    if len({gpu["index"] for gpu in devices}) != len(devices) or len({gpu["uuid"] for gpu in devices}) != len(devices):
        raise ConfigError("nvidia-smi返回重复的GPU编号或UUID，无法安全选择设备。")
    return devices

def detect_gpus(smi: str | Path | None = None, timeout: float = 10) -> list[dict]:
    executable = str(smi) if smi is not None else shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidates = [Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe",
                      Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation/NVSMI/nvidia-smi.exe"]
        executable = next((str(path) for path in candidates if path.is_file()), None)
    if not executable:
        raise ConfigError("找不到nvidia-smi。请安装或修复NVIDIA显卡驱动，重开终端后运行nvidia-smi -L；无需单独安装CUDA Toolkit。")
    command = [executable,
               "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.free,driver_version",
               "--format=csv,noheader,nounits"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired as error:
        raise ConfigError("nvidia-smi查询超时。请先运行nvidia-smi -L检查驱动响应，再重试。") from error
    except OSError as error:
        raise ConfigError(f"无法运行nvidia-smi：{error}。请修复NVIDIA驱动或检查程序路径。") from error
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()[-600:]
        raise ConfigError(f"NVIDIA设备/驱动查询失败（退出码{process.returncode}）。请运行nvidia-smi -L检查，必要时更新驱动。 {detail}")
    return parse_smi_csv(process.stdout)

def normalize_cache(cache):
    if not isinstance(cache,str): raise ConfigError('KV 类型必须为字符串。')
    cache={'q8_0':'int8_per_token_head','f16':'bfloat16','auto':'bfloat16'}.get(cache,cache)
    if cache not in CACHE_TYPE_BYTES:
        raise ConfigError('KV 必须为 int8_per_token_head、bfloat16 或 fp8_per_token_head；旧 Q4 KV 不受支持。')
    return cache

def check_gpu(gpu):
    if gpu['compute_capability'] not in ('8.0','8.6','8.9','12.0'):
        raise ConfigError('当前 Windows 运行包支持 RTX 30/40/50 对应 SM 8.0/8.6/8.9/12.0；GTX 10/16 和 RTX 20 需要另一套旧架构运行栈，暂不支持。')
    if not _version_at_least(gpu['driver_version'],'596.36'):
        raise ConfigError('本项目当前运行配置要求驱动 596.36 或更新；这是实测支持下限，不是 CUDA 官方最低版本。')


def select_profile(gpu,profile='auto'):
    check_gpu(gpu)
    if profile not in ('auto','fast','quality','compat'):
        raise ConfigError('profile 必须为 auto、fast、quality 或 compat。')
    if profile=='auto': return 'fast' if gpu['compute_capability']=='12.0' else 'compat'
    if profile in ('fast','quality') and gpu['compute_capability']!='12.0':
        raise ConfigError('NVFP4 模式限 RTX 50；请为该显卡选择自动或 INT4 兼容模式。')
    return profile

def memory_plan(*,mode='server',free_mib,model_bytes,context=2048,parallel=None,ubatch=None,cache_type_k='int8_per_token_head',cache_type_v=None,kv_gib=None):
    context=_positive_int(context,'context',32768)
    cache=normalize_cache(cache_type_k)
    if cache_type_v is not None and normalize_cache(cache_type_v)!=cache:
        raise ConfigError('vLLM K/V 必须使用相同类型。')
    if not math.isfinite(free_mib) or free_mib<=0: raise ConfigError('没有可用显存预算。')
    batch=_positive_int(ubatch or 2048,'batch tokens',32768)
    parallel=_positive_int(parallel,'parallel',256) if parallel is not None else 32
    if batch<parallel: raise ConfigError('调度 token 预算必须不小于最大并发。')
    model=math.ceil(model_bytes/MIB*1.12)
    workspace=2048
    safety=max(512,math.ceil(free_mib*.08))
    capacity=math.floor((free_mib-model-workspace-safety)/256)*256
    per_token=2*KV_ELEMENTS_PER_TOKEN*CACHE_TYPE_BYTES[cache]
    recommended=(3 if parallel<=32 else 6)*1024*(CACHE_TYPE_BYTES[cache]/2)
    kv=float(kv_gib)*1024 if kv_gib is not None else min(capacity,recommended)
    minimum=math.ceil(context/16)*16*per_token/MIB
    if not math.isfinite(kv) or kv<minimum or kv>capacity:
        raise ConfigError(f'显存预算不足或 KV 预算无效：KV 至少需 {minimum:.0f} MiB，可用 {capacity:.0f} MiB。')
    tokens=int(kv*MIB/per_token)//16*16
    kv=tokens*per_token/MIB
    return {'parallel':parallel,'context':context,'batch':batch,'ubatch':batch,'cache':cache,
        'budget':{'is_estimate':True,'free_memory_mib':free_mib,'model_reserve_mib':model,
        'workspace_reserve_mib':workspace,'safety_margin_mib':safety,'kv_total_mib':kv,
        'estimated_total_mib':model+workspace+safety+kv,'kv_token_capacity':tokens,
        'full_context_sequences':tokens//context,'kv_elements_per_token_per_cache':KV_ELEMENTS_PER_TOKEN,
        'kv_context_tokens_rounded':math.ceil(context/16)*16,
        'note':'KV 为共享 token 池；并发不是每条完整上下文的显存预留。池满时调度器排队或重算。总预算包含工作区与安全余量，实际峰值需实测。'}}

def resolve_config(root,*,mode='server',profile='auto',parallel=None,context=2048,ubatch=None,gpu=None,model_override=None,cache_type_k='int8_per_token_head',cache_type_v=None,kv_gib=None,memory_percent=75):
    from setup_model import load_manifest, checkpoint_manifest, checkpoint_path, verify_checkpoint
    root=Path(root)
    manifest=load_manifest(root/'models/manifest.json')
    if not 10<=memory_percent<=100: raise ConfigError('memory_percent 必须为 10 至 100。')
    devices=detect_gpus()
    if gpu is not None and str(gpu): devices=[d for d in devices if str(gpu) in (str(d['index']),d['uuid'])]
    errors=[]
    for device in sorted(devices,key=lambda d:-d['free_memory_mib']):
        try:
            selected=select_profile(device,profile)
            spec=checkpoint_manifest(manifest,selected)
            model=Path(model_override).resolve() if model_override else checkpoint_path(root,spec)
            verify_checkpoint(model,spec)
            limit=min(device['free_memory_mib'],device['total_memory_mib']*memory_percent/100)
            plan=memory_plan(mode=mode,free_mib=limit,model_bytes=spec['checkpoint_size_bytes'],context=context,parallel=parallel,ubatch=ubatch,cache_type_k=cache_type_k,cache_type_v=cache_type_v,kv_gib=kv_gib)
            return {**plan,'ok':True,'backend':'vllm','model':str(model),'profile':selected,
                'kernel':'cutlass' if selected=='fast' else 'marlin','gpu':device,'cuda_visible_devices':device['uuid'],
                'warnings':[] if device['compute_capability']=='12.0' else ['RTX 30/40 架构已包含在运行包；尚无对应实卡验证。']}
        except ValueError as error: errors.append(str(error))
    raise ConfigError('; '.join(errors) or '未找到选中的 NVIDIA 显卡。')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--mode',default='server')
    parser.add_argument('--profile',default='auto')
    parser.add_argument('--context',type=int,default=2048)
    parser.add_argument('--parallel',type=int)
    args=parser.parse_args()
    print(json.dumps({'backend':'vllm','gpus':detect_gpus(),'default_kv':'int8_per_token_head'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
