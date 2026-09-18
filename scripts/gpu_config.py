"""Probe NVIDIA GPUs and resolve a relocatable Hy-MT Windows configuration.

Configuration resolution only executes nvidia-smi and never loads a model.
Launchers can separately probe executable --help for quantized-cache support.
Binary compatibility comes from BIN/build-info.json. The installed bin directory
and models/manifest.json define the current project version.

Memory recommendations are estimates, not an OOM guarantee. Other GPUs have not
been performance/memory validated here. Explicit settings are never reduced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


MIB = 1024 * 1024
INT32_MAX = 2**31 - 1
PARALLEL_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MODE_EXECUTABLES = {"batch": "hy-batch.exe", "server": "llama-server.exe"}
MODEL_FILES = {
    "fast": ("Hy-MT2-1.8B-NVFP4-fused.gguf",),
    "official": ("Hy-MT2-1.8B-Q4_K_M-fused.gguf",),
}
CACHE_TYPE_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
# Hy-MT2-1.8B: 32 layers, 4 KV heads, 128 elements per head, for each of K and V.
KV_ELEMENTS_PER_TOKEN = 32 * 4 * 128


class ConfigError(ValueError):
    """An actionable configuration or device error, suitable for CLI display."""


def cache_type_args(cache_type_k: str = "f16", cache_type_v: str = "f16") -> list[str]:
    for label, value in (("cache-type-k", cache_type_k), ("cache-type-v", cache_type_v)):
        if value not in CACHE_TYPE_BYTES:
            raise ConfigError(f"{label}必须为f16、q8_0或q4_0。")
    return ["--cache-type-k", cache_type_k, "--cache-type-v", cache_type_v]


def cache_type_warnings(cache_type_k: str, cache_type_v: str) -> list[str]:
    if cache_type_k != cache_type_v:
        return ["混合K/V类型在默认构建中没有专用FlashAttention向量内核，会临时转换为f16；"
                "可能增加工作区显存并降低速度。请编译对应GGML_CUDA_FA_QUANTS组合并实测，或选择相同K/V类型。"]
    return []


def ensure_cache_type_support(executable: str | Path, cache_type_k: str, cache_type_v: str,
                              *, env: dict | None = None, cwd: str | Path | None = None) -> None:
    """Check CLI support before creating outputs; --help does not load a model."""
    cache_type_args(cache_type_k, cache_type_v)
    rebuild = ("请用当前源码运行scripts/build-source.ps1重建，选择新生成的程序目录，"
               "并确保GGML_CUDA_FA_QUANTS包含所选K/V组合。")
    try:
        result = subprocess.run([str(executable), "--help"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=15, env=env, cwd=cwd,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ConfigError(f"无法检查{executable}的量化KV支持：{error}。{rebuild}") from error
    help_text = result.stdout + result.stderr
    if result.returncode != 0 or any(flag not in help_text for flag in ("--cache-type-k", "--cache-type-v")):
        raise ConfigError(f"{executable}未声明所需的量化KV参数支持。{rebuild}")


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


def _architecture(value: str) -> tuple[tuple[int, int], str]:
    match = re.fullmatch(r"(\d{2,3})([a-z]?)", str(value))
    if not match:
        raise ConfigError(f"build-info.json中架构值无效：{value!r}。请使用随包原始元数据。")
    digits = match[1]
    return (int(digits[:-1]), int(digits[-1])), match[2]


def architecture_match(capability: str, metadata: dict) -> dict | None:
    """Exact cubins only; forward compatibility requires explicit ordinary PTX."""
    target = _capability(capability)
    if target < (7, 5):
        return None
    cubins, ptx = metadata.get("architectures"), metadata.get("ptx_architectures", [])
    if not isinstance(cubins, list) or not cubins or not isinstance(ptx, list):
        raise ConfigError("build-info.json必须声明非空architectures列表和可选ptx_architectures列表。")
    for arch in cubins:
        capability_tuple, _ = _architecture(arch)
        if capability_tuple == target:
            return {"kind": "cubin_exact", "architecture": str(arch)}
    for arch in ptx:
        capability_tuple, suffix = _architecture(arch)
        if suffix:
            raise ConfigError("架构专用PTX不能声明为通用前向兼容；请修复build-info.json。")
        if target >= capability_tuple:
            return {"kind": "ptx_forward", "architecture": str(arch)}
    return None


def _binary_candidates(root: Path, binary_dir: str | Path | None) -> list[Path]:
    if binary_dir is not None:
        path = Path(binary_dir)
        return [(path if path.is_absolute() else root / path).resolve()]
    return [root / "bin"]


def choose_binary(root: Path, mode: str, gpu: dict, binary_dir: str | Path | None = None) -> tuple[Path, dict, dict]:
    failures = []
    for directory in _binary_candidates(root, binary_dir):
        executable = directory / MODE_EXECUTABLES[mode]
        if not executable.is_file():
            failures.append(f"{executable}不存在")
            continue
        metadata_path = directory / "build-info.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as error:
                raise ConfigError(f"无法读取构建元数据{metadata_path}：{error}") from error
            if not isinstance(metadata, dict):
                raise ConfigError(f"{metadata_path}必须是JSON对象。")
            metadata = dict(metadata, metadata_source=str(metadata_path))
        else:
            failures.append(f"{directory}缺少build-info.json，不能确认GPU兼容性")
            continue
        match = architecture_match(gpu["compute_capability"], metadata)
        if match is None:
            failures.append(f"{directory}不含计算能力{gpu['compute_capability']}的已声明兼容代码")
            continue
        minimum = metadata.get("minimum_driver")
        if not isinstance(minimum, str) or not minimum:
            raise ConfigError(f"{metadata_path}未声明minimum_driver，不能确认CUDA运行时的驱动要求。")
        if not _version_at_least(gpu["driver_version"], minimum):
            failures.append(f"需要NVIDIA驱动{minimum}或更新版本，当前为{gpu['driver_version']}；请更新驱动")
            continue
        return directory.resolve(), metadata, match
    raise ConfigError("没有可用于此显卡的推理程序：" + "；".join(failures) + "。请使用包含本GPU架构和build-info.json的完整安装包。")


def choose_model(root: Path, profile: str, gpu: dict, metadata: dict,
                 model_override: str | Path | None = None) -> tuple[str, Path, list[str]]:
    supported = metadata.get("nvfp4_compute_capabilities", [])
    if not isinstance(supported, list):
        raise ConfigError("build-info.json的nvfp4_compute_capabilities必须是列表。")
    fast_allowed = gpu["compute_capability"] in supported
    if profile == "fast" and not fast_allowed:
        raise ConfigError(f"此构建未为计算能力{gpu['compute_capability']}声明NVFP4支持；请改用--profile official或auto。")
    chosen = ("fast" if fast_allowed and gpu["compute_capability"] == "12.0" else "official") if profile == "auto" else profile
    if model_override is not None:
        requested = Path(model_override)
        path = (requested if requested.is_absolute() else root / requested).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise ConfigError(f"指定模型不存在或为空：{path}。请提供已有GGUF文件路径；不会自动下载或替换模型。")
        return chosen, path, ["使用显式模型路径；profile仅表示GPU配置档，不用于推断指定模型的量化类型。"]
    try:
        manifest = json.loads((root / "models/manifest.json").read_text(encoding="utf-8-sig"))
        item = manifest["files"][chosen]
        filename, size, expected_hash = item["filename"], item["size_bytes"], item["sha256"]
        if (not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.gguf", filename)
                or type(size) is not int or size <= 0):
            raise ValueError("invalid filename or size")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError("invalid SHA-256")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ConfigError(f"无法读取当前models/manifest.json模型配置：{error}") from error
    path = root / "models" / filename
    if not path.is_file() or path.stat().st_size != size:
        raise ConfigError(f"当前模型缺失或大小与清单不符：{path}。请运行setup-model.cmd --profile {chosen}安装并校验当前模型。")
    try:
        # Always read the complete file. Same-size historical GGUFs, including
        # replacements that preserve modification time, must not pass via a
        # filename/stat cache. Explicit model_override remains user-selected.
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as error:
        raise ConfigError(f"无法读取当前模型进行SHA-256校验：{path}：{error}") from error
    if actual_hash != expected_hash.lower():
        raise ConfigError(f"当前模型SHA-256与清单不符：{path}。请运行setup-model.cmd --profile {chosen}安装并校验当前模型。")
    return chosen, path.resolve(), []


def memory_plan(*, mode: str, free_mib: float, model_bytes: int, context: int = 1024,
                parallel: int | None = None, ubatch: int | None = None,
                cache_type_k: str = "f16", cache_type_v: str = "f16") -> dict:
    """Pure conservative sizing, with every input explicit; no device or file IO."""
    if mode not in MODE_EXECUTABLES:
        raise ConfigError("mode必须为batch或server。")
    cache_type_args(cache_type_k, cache_type_v)
    context = _positive_int(context, "context")
    limit = 256 if mode == "batch" else 128
    if parallel is not None:
        _positive_int(parallel, "parallel", limit)
    if ubatch is not None:
        _positive_int(ubatch, "ubatch")
    if not math.isfinite(free_mib) or free_mib < 0 or type(model_bytes) is not int or model_bytes <= 0:
        raise ConfigError("空闲显存或模型文件大小无效。")
    aligned_context = ((context + 255) // 256) * 256
    k_per_slot = aligned_context * KV_ELEMENTS_PER_TOKEN * CACHE_TYPE_BYTES[cache_type_k] / MIB
    v_per_slot = aligned_context * KV_ELEMENTS_PER_TOKEN * CACHE_TYPE_BYTES[cache_type_v] / MIB
    kv_per_slot = k_per_slot + v_per_slot
    model_reserve = math.ceil(model_bytes / MIB * 1.10)
    small_gpu = free_mib < 6144
    workspace_base = 1024 if small_gpu else 2048
    workspace = math.ceil(workspace_base * max(1.0, (ubatch or 2048) / 2048))
    safety = 256 if small_gpu else max(256, math.ceil(free_mib * 0.05))
    fixed = model_reserve + workspace + safety
    available = max(0.0, free_mib - fixed)
    affordable = math.floor(available / kv_per_slot)
    if parallel is None:
        ceiling = min(limit, affordable, ubatch if ubatch is not None else limit,
                      INT32_MAX // context)
        choices = [value for value in PARALLEL_STEPS if value <= ceiling]
        if not choices:
            raise ConfigError(f"空闲显存{free_mib:.0f}MiB不足以容纳保守估算的单请求配置（约{fixed + kv_per_slot:.0f}MiB）。"
                              "请停止其他GPU任务、缩短--context，或使用显存更多的显卡；显式设置不会绕过预算检查。")
        selected = max(choices)
    else:
        selected = parallel
        if selected * context > INT32_MAX:
            raise ConfigError("parallel×context超过int32上限，请降低并发或上下文。")
        if selected > affordable:
            raise ConfigError(f"显式parallel={selected}预计需{fixed + selected * kv_per_slot:.0f}MiB，当前仅空闲{free_mib:.0f}MiB。"
                              f"请降低--parallel（预算上限约{max(0, affordable)}）或--context，或先停止其他GPU任务；未自动修改显式参数。")
    microbatch = ubatch
    if microbatch is None:
        microbatch = 256 if small_gpu else max(256, min(2048 if mode == "batch" else 512, selected * 8))
    if microbatch < selected:
        raise ConfigError("ubatch不能小于parallel；请增大--ubatch或降低--parallel。")
    kv_total = selected * kv_per_slot
    return {"parallel": selected, "context": context, "ubatch": microbatch, "batch": max(2048, microbatch),
            "cache_type_k": cache_type_k, "cache_type_v": cache_type_v,
            "budget": {"is_estimate": True, "free_memory_mib": free_mib,
                       "model_file_bytes": model_bytes, "model_reserve_mib": model_reserve,
                       "workspace_reserve_mib": workspace, "safety_margin_mib": safety,
                       "kv_context_tokens_rounded": aligned_context, "kv_per_slot_mib": kv_per_slot,
                       "k_per_slot_mib": k_per_slot, "v_per_slot_mib": v_per_slot,
                       "k_total_mib": selected * k_per_slot, "v_total_mib": selected * v_per_slot,
                       "kv_elements_per_token_per_cache": KV_ELEMENTS_PER_TOKEN,
                       "kv_total_mib": kv_total, "estimated_total_mib": fixed + kv_total,
                       "remaining_after_estimate_mib": free_mib - fixed - kv_total,
                       "parallel_was_explicit": parallel is not None, "ubatch_was_explicit": ubatch is not None,
                       "note": "按Hy-MT2-1.8B结构及查询时空闲显存估算，不保证不会OOM；量化KV包含块缩放开销，实际性能/质量须实测。"}}


def resolve_config(root: str | Path, mode: str = "batch", profile: str = "auto",
                   parallel: int | None = None, context: int = 1024, ubatch: int | None = None,
                   gpu: str | int | None = None, binary_dir: str | Path | None = None,
                   model_override: str | Path | None = None,
                   cache_type_k: str = "f16", cache_type_v: str = "f16") -> dict:
    root = Path(root).expanduser().resolve()
    if mode not in MODE_EXECUTABLES or profile not in ("auto", "fast", "official"):
        raise ConfigError("mode必须为batch/server，profile必须为auto/fast/official。")
    cache_type_args(cache_type_k, cache_type_v)
    _positive_int(context, "context")
    if parallel is not None:
        _positive_int(parallel, "parallel", 256 if mode == "batch" else 128)
    if ubatch is not None:
        _positive_int(ubatch, "ubatch")
    devices = detect_gpus()
    if gpu is not None:
        selector = str(gpu).strip()
        devices = [item for item in devices if selector == str(item["index"]) or selector == item["uuid"]]
        if not devices:
            raise ConfigError(f"没有找到指定GPU {selector!r}；请运行nvidia-smi -L，传入实际编号或完整GPU UUID。")
    devices.sort(key=lambda item: (-item["free_memory_mib"], item["index"]))
    failures = []
    for selected in devices:
        try:
            if _capability(selected["compute_capability"]) < (7, 5):
                raise ConfigError("此CUDA 13安装包要求计算能力至少7.5（GTX16/RTX20及支持清单内更新显卡）。")
            directory, metadata, match = choose_binary(root, mode, selected, binary_dir)
            chosen_profile, model, warnings = choose_model(root, profile, selected, metadata, model_override)
            warnings += cache_type_warnings(cache_type_k, cache_type_v)
            plan = memory_plan(mode=mode, free_mib=selected["free_memory_mib"], model_bytes=model.stat().st_size,
                               context=context, parallel=parallel, ubatch=ubatch,
                               cache_type_k=cache_type_k, cache_type_v=cache_type_v)
            return {"ok": True, "root": str(root), "mode": mode, "profile_requested": profile,
                    "profile": chosen_profile, "model": str(model), "binary_dir": str(directory),
                    "model_override": str(model) if model_override is not None else None,
                    "executable": str(directory / MODE_EXECUTABLES[mode]), "gpu": selected,
                    "gpu_index": selected["index"], "gpu_uuid": selected["uuid"], "device_index": 0,
                    "cuda_visible_devices": selected["uuid"], "environment": {"CUDA_VISIBLE_DEVICES": selected["uuid"]},
                    "architecture_match": match, "build_info": metadata,
                    "sampling_threads": 1 if mode == "batch" else 4, "gpu_prefix": mode == "batch",
                    "warnings": warnings, **plan}
        except ConfigError as error:
            failures.append(f"GPU {selected['index']}（{selected['name']}）：{error}")
    raise ConfigError("未找到可运行的GPU配置。" + "\n".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=MODE_EXECUTABLES, default="batch")
    parser.add_argument("--profile", choices=("auto", "fast", "official"), default="auto")
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--ubatch", type=int)
    parser.add_argument("--cache-type-k", choices=CACHE_TYPE_BYTES, default="f16")
    parser.add_argument("--cache-type-v", choices=CACHE_TYPE_BYTES, default="f16")
    parser.add_argument("--gpu", help="nvidia-smi设备编号或完整GPU UUID")
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--model", type=Path, help="仅使用指定现存GGUF，按实际文件大小估算显存")
    parser.add_argument("--json", action="store_true", help="ASCII JSON，兼容PowerShell 5.1捕获")
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args.root, mode=args.mode, profile=args.profile, parallel=args.parallel,
                                context=args.context, ubatch=args.ubatch, gpu=args.gpu, binary_dir=args.binary_dir,
                                model_override=args.model, cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v)
    except (ConfigError, OSError) as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=True))
        else:
            print(f"配置错误：{error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(config, ensure_ascii=True))
    else:
        device = config["gpu"]
        print(f"GPU {device['index']}: {device['name']}，计算能力{device['compute_capability']}，空闲显存{device['free_memory_mib']:.0f}MiB")
        print(f"模型：{config['model']}\n程序：{config['executable']}")
        print(f"并发{config['parallel']}，每条上下文{config['context']}，ubatch={config['ubatch']}，GPU前缀={config['gpu_prefix']}")
        print(f"KV缓存：K={config['cache_type_k']}，V={config['cache_type_v']}，估算{config['budget']['kv_total_mib']:.1f}MiB")
        print(config["budget"]["note"])
        for warning in config["warnings"]:
            print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
