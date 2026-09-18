"""Export allowlisted upstream/native benchmark evidence without request history.

No inference is performed. Reports must contain all requested repetitions.
Quantization may differ between the two implementations and is always recorded
separately. File hashes identify artifacts; HTTP JSONL contains timing fields,
so its whole-file hash is not a translation-equality claim.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
METRICS = """wall_s initialization_s loaded_process_wall_s
process_wall_s_including_initialization output_tokens output_tokens_per_second
successful_requests errors truncated_requests complete_translations
complete_output_tokens complete_output_tokens_per_second complete_translations_per_second
text_tokens_without_eog text_tokens_without_eog_per_second dummy_decode_tokens
graphs_reused sampling_threads valid_throughput_run""".split()
NATIVE_CONFIG = """parallel context_per_sequence max_tokens batch ubatch seed greedy
sampling_threads batch_snapshot backend_sampling kv_unified gpu_prefix continuous
refill_min pad_final_wave_effective temperature top_p top_k min_p repeat_penalty
repeat_last_n""".split()
WORKLOAD = """dataset_sha256 requests parallel context_per_sequence batch ubatch
cpu_threads seed_rule cache_prompt warmup repeats""".split()
SAMPLING = "temperature top_p top_k repeat_penalty max_tokens seed greedy min_p repeat_last_n".split()
HARDWARE = "gpu vram_gb cpu os nvidia_driver cuda_toolkit msvc".split()
BUILD_INFO = """platform cuda_toolkit architectures ptx_architectures cpu_baseline
ggml_native cpu_avx cpu_avx2 cpu_avx512 source_commit source_dirty build_type""".split()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pick(value: dict, keys: list[str]) -> dict:
    # Keys are explicitly scoped at every nesting level. Never copy a report,
    # command, environment, hardware object, or request object wholesale.
    return {key: value[key] for key in keys if key in value}


def checked_sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError("Invalid SHA256 in benchmark metadata")
    return value.lower()


def binary_hashes(value: dict) -> dict:
    result = {}
    for name, sha in value.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:exe|dll)", name, re.IGNORECASE):
            raise ValueError("Binary hash key must be a filename, not a path")
        result[name] = checked_sha(sha)
    return result


def basename(value: str) -> str:
    return PureWindowsPath(value.replace("/", "\\")).name


def model(value: dict) -> dict:
    return {"filename": basename(value["filename"]), "sha256": checked_sha(value["sha256"])}


def command_configuration(command: list[str]) -> dict:
    # The raw command contains model/user paths and must never be published.
    options = {
        "-ngl": "gpu_layers", "-fa": "flash_attention", "-c": "total_context",
        "-np": "parallel", "-b": "batch", "-ub": "ubatch", "-t": "cpu_threads",
        "-tb": "cpu_batch_threads", "--temp": "temperature", "--top-p": "top_p",
        "--top-k": "top_k", "--min-p": "min_p", "--repeat-penalty": "repeat_penalty",
        "--repeat-last-n": "repeat_last_n", "-ctk": "kv_type_k", "-ctv": "kv_type_v",
    }
    result = {}
    for index, token in enumerate(command):
        if token in options:
            if index + 1 == len(command):
                raise ValueError(f"Incomplete baseline option: {token}")
            value = command[index + 1]
            if re.fullmatch(r"-?\d+", value):
                value = int(value)
            elif re.fullmatch(r"-?\d+\.\d+", value):
                value = float(value)
            elif value not in ("all", "on", "off", "auto", "f16", "f32", "q8_0", "q4_0"):
                raise ValueError(f"Unsupported baseline option value: {token}")
            result[options[token]] = value
    result["jinja"] = "--jinja" in command
    result["startup_warmup"] = "--no-warmup" not in command
    return result


def row_statistics(rows: list[dict], seed: int) -> dict:
    counts = [row.get("completion_tokens") for row in rows]
    valid_counts = all(type(value) is int and value > 0 for value in counts)
    return {
        "request_count": len(rows),
        "errors": sum(not row.get("ok", False) for row in rows),
        "truncated_requests": sum(row.get("finish_reason") == "length" for row in rows),
        "empty_outputs": sum(not str(row.get("translation", "")).strip() for row in rows),
        "stopped_requests": sum(row.get("finish_reason") == "stop" for row in rows),
        "all_completion_counts_present_and_positive": valid_counts,
        "completion_tokens_sum": sum(counts) if valid_counts else None,
        "seed_sequence_matches": all(row.get("seed") == seed + index for index, row in enumerate(rows)),
    }


def validate_metrics(metrics: dict, requests: int, rows: dict | None) -> bool:
    wall, tokens, tps = (metrics[key] for key in ("wall_s", "output_tokens", "output_tokens_per_second"))
    if (type(tokens) is not int or tokens <= 0 or not math.isfinite(wall) or wall <= 0 or
            not math.isfinite(tps) or not math.isclose(tokens / wall, tps, rel_tol=1e-8)):
        raise ValueError("Token count, wall clock and throughput disagree")
    valid = (metrics.get("valid_throughput_run") is True and metrics.get("errors") == 0 and
             metrics.get("truncated_requests") == 0 and metrics.get("successful_requests") == requests)
    if rows is not None:
        valid = (valid and rows["request_count"] == requests and rows["errors"] == 0 and
                 rows["truncated_requests"] == rows["empty_outputs"] == 0 and
                 rows["stopped_requests"] == requests and rows["seed_sequence_matches"] and
                 rows["completion_tokens_sum"] == tokens)
    return valid


def distribution(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values}


def gpu_observation(text: str) -> list[dict]:
    devices = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("Unexpected six-column GPU observation")
        device = {"name": row[0].strip(), "driver": row[1].strip()}
        for name, value in zip(("total_memory_mib", "used_memory_mib", "utilization_percent", "temperature_c"), row[2:]):
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)(?:\s*(?:MiB|%))?\s*", value)
            device[name] = float(match[1]) if match else None
        devices.append(device)
    return devices


def clean_release(release: dict) -> dict:
    result = pick(release, ["upstream_repository", "requested_tag", "requested_commit"])
    for name, keys in {
        "requested_release": ["url", "workflow_url", "workflow_conclusion", "windows_cuda_binary_available_at_check", "checked_date"],
        "selected_release": ["tag", "commit", "release_url", "published_utc"],
        "binary_asset": ["filename", "url", "size_bytes", "sha256", "github_asset_digest_verified", "extracted_files", "binaries_modified"],
        "version_check": ["executable", "arguments", "exit_code", "output", "help_exit_code"],
        "build": ["official_cuda_toolkit", "official_cpu_compiler_reported", "official_cpu_all_variants", "official_workflow_url",
                  "custom_cuda_toolkit", "custom_cpu_compiler", "custom_cpu_baseline"],
        "source_comparison": ["selected_release_is_direct_parent", "inference_source_identical", "changed_paths", "commit_evidence_url", "comparison_url"],
    }.items():
        if name in release:
            result[name] = pick(release[name], keys)
    dependencies = release.get("runtime_dependency_check", {})
    result["runtime_dependency_check"] = pick(dependencies, ["ggml_cuda_load_succeeded", "gpu_inference_run_by_this_check", "dynamic_cudart_dll_observed"])
    result["runtime_dependency_check"]["loaded_cuda_dependencies"] = [
        pick(item, ["filename", "file_version", "product_version", "runtime_distribution"])
        for item in dependencies.get("loaded_cuda_dependencies", [])
    ]
    return result


def clean_report(path: Path) -> dict:
    report = read_json(path)
    models = report["models"]
    stock_quant = models.get("stock_quantization", models.get("quantization"))
    optimized_quant = models.get("optimized_quantization", models.get("quantization"))
    if stock_quant not in ("NVFP4", "Q4_K_M") or optimized_quant not in ("NVFP4", "Q4_K_M"):
        raise ValueError("Both model quantizations must be identified")
    workload = pick(report["workload"], WORKLOAD)
    workload["sampling"] = pick(report["workload"]["sampling"], SAMPLING)
    baseline = pick(report["baseline"], ["source_commit", "tag", "version"])
    baseline["binary_sha256"] = binary_hashes(report["baseline"]["binary_sha256"])
    baseline["configuration"] = command_configuration(report["baseline"]["command"])
    runs = []
    expected_repeats = list(range(1, workload["repeats"] + 1))
    for implementation in ("upstream", "optimized"):
        actual = sorted(row["repeat"] for row in report["runs"] if row["implementation"] == implementation)
        if actual != expected_repeats:
            raise ValueError(f"Report is incomplete or repeats are duplicated: {path.name}: {implementation}")
    if len(report["runs"]) != 2 * workload["repeats"]:
        raise ValueError("Unexpected benchmark implementation")
    for run in report["runs"]:
        implementation, repeat = run["implementation"], run["repeat"]
        clean = {"implementation": implementation, "repeat": repeat}
        rows = None
        if implementation == "upstream":
            metrics = pick(run["metrics"], METRICS)
            clean["output_jsonl_sha256"] = checked_sha(run["output_jsonl_sha256"])
            output = path.parent / f"stock-{repeat}.jsonl"
            if output.is_file():
                if file_sha256(output) != clean["output_jsonl_sha256"]:
                    raise ValueError("Upstream JSONL hash changed")
                rows = [json.loads(line) for line in output.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            native = run["native_report"]
            if native["dataset_sha256"] != workload["dataset_sha256"] or native["cases"] != workload["requests"]:
                raise ValueError("Native dataset differs from paired upstream workload")
            metrics = pick(native["summary"], METRICS)
            clean["configuration"] = pick(native["native_summary"], NATIVE_CONFIG)
            for native_key, shared_key in (("parallel", "parallel"), ("context_per_sequence", "context_per_sequence"),
                                           ("batch", "batch"), ("ubatch", "ubatch")):
                if clean["configuration"][native_key] != workload[shared_key]:
                    raise ValueError(f"Native configuration differs from shared workload: {native_key}")
            if native["sampling"] != workload["sampling"]:
                raise ValueError("Native sampling differs from paired upstream sampling")
            clean["source_report_sha256"] = checked_sha(run["source_report_sha256"])
            name = basename(run["source_report"])
            output = path.parent.parent / Path(name).with_suffix(".jsonl")
            if not output.is_file():
                raise FileNotFoundError(f"Native output required for artifact hash: {output}")
            clean["output_jsonl_sha256"] = file_sha256(output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            report_rows = native.get("results", [])
            match_keys = ["id", "seed", "translation", "finish_reason", "completion_tokens", "output_tokens", "ok"]
            if [pick(row, match_keys) for row in rows] != [pick(row, match_keys) for row in report_rows]:
                raise ValueError("Native output file differs from embedded report results")
        clean["metrics"] = metrics
        checks = row_statistics(rows, workload["sampling"]["seed"]) if rows is not None else None
        clean["independent_row_checks"] = checks
        clean["valid"] = validate_metrics(metrics, workload["requests"], checks)
        runs.append(clean)
    complete_valid = all(run["valid"] for run in runs)
    statistics_by_implementation = {}
    for implementation in ("upstream", "optimized"):
        subset = [run for run in runs if run["implementation"] == implementation]
        statistics_by_implementation[implementation] = {
            "runs": len(subset), "all_valid": all(run["valid"] for run in subset),
            "completion_tps": distribution([run["metrics"]["output_tokens_per_second"] for run in subset]),
            "wall_s": distribution([run["metrics"]["wall_s"] for run in subset]),
        }
    upstream, optimized = (statistics_by_implementation[key] for key in ("upstream", "optimized"))
    result = {
        "source_report_sha256": file_sha256(path),
        "baseline_quantization": stock_quant, "optimized_quantization": optimized_quant,
        "quantization_differs": stock_quant != optimized_quant,
        "baseline": baseline, "models": {key: model(models[key]) for key in ("stock", "optimized")},
        "workload": workload, "runs": runs, "summary": statistics_by_implementation,
        "all_valid": complete_valid,
        "median_tps_ratio": optimized["completion_tps"]["median"] / upstream["completion_tps"]["median"] if complete_valid else None,
        "median_wall_ratio": upstream["wall_s"]["median"] / optimized["wall_s"]["median"] if complete_valid else None,
    }
    if "optimized" in report:
        result["optimized"] = pick(report["optimized"], ["source_commit", "version", "binary_hash_capture"])
        if "binary_sha256" in report["optimized"]:
            result["optimized"]["binary_sha256"] = binary_hashes(report["optimized"]["binary_sha256"])
    if "optimized_binary_sha256" in report:
        result["optimized"] = {
            "binary_sha256": binary_hashes(report["optimized_binary_sha256"]),
            "build": pick(report.get("optimized_build", {}), BUILD_INFO),
            "binary_hash_capture": "Recorded in the benchmark report before its repetitions",
        }
    if "hardware" in report:
        result["hardware"] = pick(report["hardware"], HARDWARE + ["cpu_identifier"])
    if report.get("gpu_before_measurement"):
        result["gpu_before_measurement"] = gpu_observation(report["gpu_before_measurement"])
    return result


def assert_no_private_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            assert_no_private_strings(key)
            assert_no_private_strings(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_private_strings(child)
    elif isinstance(value, str):
        if re.search(r"(?i)(?:^[a-z]:[/\\]|[a-z]:\\|\\\\|/Users/|/home/|GPU-[0-9a-f-]{8,}|pid_\d+)", value):
            raise ValueError("Private path or device/process identifier in allowlisted field")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/upstream-comparison.json")
    parser.add_argument("--hardware", type=Path, default=ROOT / "benchmarks/rtx5090-windows.json")
    parser.add_argument("--upstream-release", type=Path, default=ROOT / "benchmarks/upstream-release.json")
    parser.add_argument("--optimized-bin", type=Path, default=ROOT / "bin")
    parser.add_argument("--unpack-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    comparisons = [clean_report(path.resolve()) for path in args.report]
    hardware_file = read_json(args.hardware)
    captured_hardware = comparisons[0].get("hardware")
    result = {
        "schema_version": 1, "summarized_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": captured_hardware or pick(hardware_file.get("hardware", hardware_file), HARDWARE),
        "hardware_source": ("First benchmark report; per-report GPU observations are retained separately"
                            if captured_hardware else "Existing hardware metadata fallback for legacy reports; not live detection"),
        "comparisons": comparisons,
        "measurement_boundaries": {
            "scope": "Complete deployed translation systems; untouched upstream HTTP versus optimized native JSONL. Quantization may differ.",
            "upstream_wall": "HTTP client execution through completion, including request queue and first inference; excludes model startup and post-run JSONL write.",
            "native_wall": "Input read, prompt processing, generation, sampling and JSONL write; includes first graph work, excludes model/context initialization.",
            "upstream_loaded_process_wall": "Start to final HTTP response, before server termination and JSONL write; not directly equivalent to native subprocess start-to-exit time.",
            "tokens": "Completion counts include actual EOG; native padding tokens excluded, padding work included in wall time.",
            "order": "Upstream then optimized in each repetition; fixed order, not randomized.",
            "hashes": "Output hashes identify artifacts, not cross-interface translation equality; HTTP JSONL includes timings.",
            "limits": "Source revisions, compilers, interface and scheduling differ. Ratios are not isolated source-code or kernel speedups, nor a translation-quality score.",
        },
    }
    build_path = args.optimized_bin / "build-info.json"
    result["optimized_artifacts_at_summary_time"] = {
        "capture_boundary": "Files hashed when summarizing; not a per-run binary capture",
        "binary_sha256": binary_hashes({path.name: file_sha256(path) for path in sorted(args.optimized_bin.iterdir())
                                         if path.is_file() and path.suffix.lower() in (".exe", ".dll")}),
        "build": pick(read_json(build_path), BUILD_INFO) if build_path.is_file() else {},
    }
    if args.upstream_release.is_file():
        release = read_json(args.upstream_release)
        result["upstream_release"] = clean_release(release)
        result["upstream_release"]["metadata_sha256"] = file_sha256(args.upstream_release)
    proofs = []
    for path in args.unpack_manifest:
        proof = pick(read_json(path), ["input_sha256", "output_sha256", "output_bytes", "quantization_changed",
                                      "all_tensor_slices_verified", "metadata_verified_except_removed_key",
                                      "tensor_order", "packed_tensors", "output_tensors", "verified_tensor_bytes",
                                      "expected_output_sha256", "whole_file_matches_expected"])
        proof["manifest_sha256"] = file_sha256(path)
        proofs.append(proof)
    result["unpack_verification"] = proofs
    assert_no_private_strings(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": len(comparisons),
                      "all_valid": all(item["all_valid"] for item in comparisons)}, ensure_ascii=True))
    return 0 if all(item["all_valid"] for item in comparisons) else 2


if __name__ == "__main__":
    raise SystemExit(main())
