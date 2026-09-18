"""Compare an untouched upstream llama-server release with the shipped batch engine.

Uses identical prompts, seeds and concurrency. Quantization may differ when
--optimized-quantization is selected explicitly. This measures the complete
translation paths (HTTP versus native JSONL), not isolated
CUDA kernels. No source patch is applied to the upstream binary. Each measured
run starts a fresh process; initialization is reported separately on both sides.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from run_native_experiment import build_requests, ROOT
from translate import Sampling, request_json, translate_many


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-dir", type=Path, required=True)
    parser.add_argument("--stock-tag", default="b11029")
    parser.add_argument("--stock-commit", default="5c53396b89b05666c9d57445b7616a35f6198a16")
    parser.add_argument("--stock-model", type=Path, required=True)
    parser.add_argument("--optimized-model", type=Path, required=True)
    parser.add_argument("--quantization", choices=("NVFP4", "Q4_K_M"), required=True)
    parser.add_argument("--optimized-quantization", choices=("NVFP4", "Q4_K_M"))
    parser.add_argument("--parallel", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    optimized_quantization = args.optimized_quantization or args.quantization
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label):
        parser.error("Invalid label")
    if not 1 <= args.parallel <= 256 or args.repeats < 1 or args.rounds < 1:
        parser.error("Invalid concurrency or repetitions")
    stock_dir = args.stock_dir.resolve()
    stock_model = args.stock_model.resolve()
    optimized_model = args.optimized_model.resolve()
    exe = stock_dir / "llama-server.exe"
    for path in (exe, stock_model, optimized_model, ROOT / "bin/hy-batch.exe"):
        if not path.is_file():
            raise FileNotFoundError(path)
    destination = ROOT / "results" / args.label
    destination.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("LLAMA_", "GGML_", "CUDA_VISIBLE")):
            del env[key]
    env["PATH"] = os.pathsep.join((str(stock_dir), str(ROOT / "runtime/cuda"),
                                   str(ROOT / "runtime/msvc"), env.get("PATH", "")))
    env["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
    version = subprocess.run([str(exe), "--version"], env=env, capture_output=True,
                             timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    version_text = (version.stdout + version.stderr).decode("utf-8", "replace")
    if version.returncode or args.stock_commit[:8] not in version_text:
        raise RuntimeError("Upstream binary version differs from --stock-commit")
    dataset, dataset_bytes, requests = build_requests(ROOT / "scripts/benchmark_cases.json", 64, args.rounds)
    sampling = Sampling()
    command = [str(exe), "-m", str(stock_model), "--alias", "hy-mt2", "--host", "127.0.0.1",
               "--port", str(args.port), "-ngl", "all", "-fa", "on", "-c", str(args.parallel * 1024),
               "-np", str(args.parallel), "-b", "2048", "-ub", "2048", "-t", "8", "-tb", "8",
               "--jinja", "--temp", "0.7", "--top-p", "0.6", "--top-k", "20", "--min-p", "0",
               "--repeat-penalty", "1.05", "--repeat-last-n", "4096", "--no-warmup"]
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"os": platform.platform(), "cpu_identifier": platform.processor()},
        "baseline": {"source_commit": args.stock_commit, "tag": args.stock_tag,
                     "version": version_text, "command": command,
                     "binary_sha256": {path.name: digest(path) for path in sorted(stock_dir.iterdir())
                                       if path.suffix.lower() in (".exe", ".dll")}},
        "models": {"quantization": args.quantization,
                   "stock_quantization": args.quantization, "optimized_quantization": optimized_quantization,
                   "stock": {"filename": stock_model.name, "sha256": digest(stock_model)},
                   "optimized": {"filename": optimized_model.name, "sha256": digest(optimized_model)}},
        "workload": {"dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                     "requests": len(requests), "parallel": args.parallel, "context_per_sequence": 1024,
                     "batch": 2048, "ubatch": 2048, "cpu_threads": 8,
                     "sampling": asdict(sampling), "seed_rule": "42 + request index",
                     "cache_prompt": False, "warmup": "none; first graph work included",
                     "repeats": args.repeats},
        "optimized_binary_sha256": {path.name: digest(path) for path in sorted((ROOT / "bin").iterdir())
                                    if path.suffix.lower() in (".exe", ".dll")},
        "optimized_build": json.loads((ROOT / "bin/build-info.json").read_text(encoding="utf-8")),
        "measurement": {"scope": "complete translation pipeline; upstream HTTP vs optimized native JSONL",
                        "wall": "initialization excluded; prompt processing and generation included; HTTP includes client queue but excludes JSONL write; native includes JSONL read/write",
                        "process_times": "stock loaded_process_wall_s ends before shutdown/output write; native process wall includes process exit; not directly comparable",
                        "tokens": "completion tokens including EOG; native dummy tokens excluded",
                        "order": "stock then optimized, fresh processes for every pair"},
        "runs": [],
    }
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=30,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    if gpu.returncode:
        raise RuntimeError("Cannot record current GPU state")
    report["gpu_before_measurement"] = gpu.stdout.strip()
    report["hardware"]["nvidia_smi"] = gpu.stdout.strip()
    save(destination / "report.json", report)
    for repeat in range(1, args.repeats + 1):
        print(f"Run {repeat}/{args.repeats}: upstream {args.quantization} p{args.parallel}", flush=True)
        with (destination / f"stock-{repeat}.stdout.log").open("wb") as out, \
             (destination / f"stock-{repeat}.stderr.log").open("wb") as err:
            start = time.perf_counter()
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Upstream server exited: {process.returncode}; inspect stderr")
                    try:
                        if request_json(f"http://127.0.0.1:{args.port}/health", None, 1).get("status") == "ok":
                            break
                    except (OSError, ValueError, RuntimeError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Upstream server did not become ready")
                    time.sleep(.25)
                initialization = time.perf_counter() - start
                rows, wall = translate_many(requests, url=f"http://127.0.0.1:{args.port}", model="hy-mt2",
                                            sampling=sampling, concurrency=args.parallel, timeout=300, cache_prompt=False)
                end_to_end = time.perf_counter() - start
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=30)
        output = destination / f"stock-{repeat}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        errors = sum(not row["ok"] for row in rows)
        truncated = sum(row["truncated"] for row in rows)
        tokens = sum(row["completion_tokens"] or 0 for row in rows)
        valid = len(rows) == len(requests) and errors == truncated == 0 and tokens > 0
        metrics = {"wall_s": wall, "initialization_s": initialization, "loaded_process_wall_s": end_to_end,
                   "output_tokens": tokens, "output_tokens_per_second": tokens / wall,
                   "successful_requests": len(rows) - errors, "errors": errors,
                   "truncated_requests": truncated, "valid_throughput_run": valid}
        report["runs"].append({"implementation": "upstream", "repeat": repeat, "metrics": metrics,
                               "output_jsonl_sha256": digest(output)})
        save(destination / "report.json", report)
        if not valid:
            raise RuntimeError("Invalid upstream throughput run; inspect request results")
        print(f"Upstream: {tokens / wall:.2f} completion token/s; {wall:.3f} s", flush=True)
        print(f"Run {repeat}/{args.repeats}: optimized {optimized_quantization} p{args.parallel}", flush=True)
        native_label = f"{args.label}-native-{repeat}"
        native_command = [sys.executable, "-E", "-s", str(ROOT / "scripts/run_native_experiment.py"),
                          "--binary", "bin/hy-batch.exe", "--model", str(optimized_model), "--parallel", str(args.parallel),
                          "--context", "1024", "--batch", "2048", "--ubatch", "2048", "--sampling-threads", "1",
                          "--rounds", str(args.rounds), "--eager-graphs", "--gpu-prefix", "--label", native_label]
        with (destination / f"native-{repeat}.runner.log").open("wb") as log:
            result = subprocess.run(native_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=1800, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError(f"Native run failed ({result.returncode}); inspect runner log")
        native_report = ROOT / "results" / f"{native_label}.json"
        native = json.loads(native_report.read_text(encoding="utf-8-sig"))
        report["runs"].append({"implementation": "optimized", "repeat": repeat,
                               "source_report": native_report.name, "source_report_sha256": digest(native_report),
                               "native_report": native})
        save(destination / "report.json", report)
    print(f"Completed: results/{args.label}/report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
