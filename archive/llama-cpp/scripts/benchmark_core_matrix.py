"""Interleaved native-engine comparison; Windows, pretokenized, no HTTP.

Stop other inference services first. Dot-source vllm_windows_env.ps1 so both
MSVC JIT and the native model registration are available. Use a fresh output
directory. This runner does not stop or start production services itself.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from benchmark_inputs import ROOT, build_requests


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@contextmanager
def telemetry(path):
    fields = ["clocks.sm", "clocks.mem", "power.draw", "utilization.gpu",
              "memory.used", "temperature.gpu", "pstate"]
    proc = subprocess.Popen(["nvidia-smi", "--id=0", "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    ready = threading.Event()
    def reader():
        with path.open("w", encoding="utf-8") as stream:
            for line in proc.stdout:
                row = dict(zip(fields, [x.strip() for x in line.split(",")]))
                row["received_unix_s"] = time.time()
                stream.write(json.dumps(row) + "\n")
                ready.set()
    worker = threading.Thread(target=reader)
    worker.start()
    try:
        if not ready.wait(15):
            raise RuntimeError("GPU telemetry did not start")
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        worker.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1 or args.repeats < 1:
        parser.error("Trials and repeats must be positive")
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    _, raw, cases = build_requests(ROOT / "scripts/benchmark_cases.json", 64, 8)
    inputs = dest / "input.jsonl"
    inputs.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cases), encoding="utf-8")
    env = {k:v for k,v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_", "CUDA_VISIBLE"))}
    env.update(PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="0", CUDA_CACHE_PATH=str(ROOT / "cache/cuda"),
        LLAMA_HYMT_FUSED_PROJ="1", GGML_CUDA_HYMT_DISABLE_ROPE_NORM="0",
        GGML_CUDA_HYMT_ROPE_NORM_STRICT="1", LLAMA_HYMT_SPARSE_PENALTIES="1",
        GGML_CUDA_HYMT_DISABLE_TOPK="0", LLAMA_HYMT_SAMPLING_SNAPSHOT="1",
        LLAMA_HYMT_BATCH_SNAPSHOT="1", GGML_CUDA_HYMT_EAGER_GRAPHS="1",
        GGML_CUDA_GRAPH_OPT="0", GGML_CUDA_HYMT_DISABLE_Q8_KV_FUSION="0",
        GGML_CUDA_Q8_KV_SUBWARP="0", HYMT_CORE_REPEATS=str(args.repeats),
        HYMT_VLLM_NATIVE="1", VLLM_HOST_IP="127.0.0.1")
    env["PATH"] = os.pathsep.join([str(ROOT / "bin"), str(ROOT / "runtime/cuda"), str(ROOT / "runtime/msvc"), env["PATH"]])
    save(dest / "protocol.json", {"dataset_sha256":hashlib.sha256(raw).hexdigest(),
        "requests":512, "trials":args.trials, "warm_repeats":args.repeats,
        "environment":{k:v for k,v in env.items() if k.startswith(("LLAMA_","GGML_","HYMT_"))},
        "metric":"Pretokenized native engine wall; includes scheduling, prefill, sampling, decode and detokenization; excludes model init, input parsing/tokenization and output JSON writing."})
    for trial in range(1, args.trials + 1):
        parallel_order = (32,256) if trial % 2 else (256,32)
        backends = ("llama","vllm") if trial % 2 else ("vllm","llama")
        for parallel in parallel_order:
            for backend in backends:
                name = f"{backend}-p{parallel}-t{trial}"
                out = dest / name
                if backend == "llama":
                    out.mkdir()
                    command = [str(args.native.resolve()), "--model", str(ROOT / "models/Hy-MT2-1.8B-NVFP4-fused.gguf"),
                        "--input",str(inputs),"--output",str(out / "output"),"--summary",str(out / "summary"),
                        "--parallel",str(parallel),"--context","1024","--batch-size","2048","--ubatch-size","2048",
                        "--max-tokens","512","--threads","8","--seed","42","--sampling-threads","1","--gpu-prefix",
                        "--cache-type-k","f16","--cache-type-v","f16"]
                else:
                    cfg = {"model":str(ROOT / "models/Hy-MT2-1.8B-NVFP4-vllm"),"dtype":"bfloat16",
                        "max_model_len":1024,"max_num_seqs":parallel,"max_num_batched_tokens":2048,
                        "enable_prefix_caching":False,"kv_cache_memory_bytes":(3 if parallel==32 else 6)*1024**3,
                        "disable_log_stats":True,"kernel_config":{"linear_backend":"cutlass","enable_flashinfer_autotune":False}}
                    config = dest / (name + ".config.json")
                    save(config, cfg)
                    command = [str(ROOT / ".local/vllm-win/Scripts/python.exe"),str(ROOT / "scripts/benchmark_vllm_offline.py"),
                        "--config",str(config),"--input",str(inputs),"--pretokenized","--repeats",str(args.repeats),"--output",str(out)]
                save(dest / (name + ".command.json"), command)
                with telemetry(dest / (name + ".gpu.jsonl")), (dest / (name + ".log")).open("wb") as log:
                    proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        code = proc.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"], capture_output=True)
                        raise
                if code:
                    raise RuntimeError(f"{name}: exit {code}; inspect log")
                if backend == "llama":
                    rows=[json.loads((out / f"summary.warm-{i}.json").read_text(encoding="utf-8")) for i in range(1,args.repeats+1)]
                    rate=statistics.median(r["core_completion_tokens_per_second"] for r in rows)
                else:
                    report=json.loads((out / "report.json").read_text(encoding="utf-8"))
                    rate=report["warm_median"]["completion_tokens_per_second"]
                print(name, round(rate,2), flush=True)


if __name__ == "__main__":
    main()
