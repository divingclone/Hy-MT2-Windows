"""Run the fixed-wave Hy-MT JSONL executable on the HTTP benchmark's exact cases.

No HTTP service is started. Native wall time includes first-use graph warm-up,
JSONL reading/writing, templates, tokenization, prefill, sampling, and decode.
Model/context initialization is reported separately. No dummy token contributes
to output throughput. Completion-token counts include a terminal EOG token,
matching the convention used for the HTTP benchmark's completion_tokens.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from translate import translation_prompt
from gpu_config import CACHE_TYPE_BYTES, cache_type_args, cache_type_warnings, ensure_cache_type_support


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def build_requests(dataset: Path, count: int, rounds: int) -> tuple[dict, bytes, list[dict]]:
    raw = dataset.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    cases = data["cases"][:count]
    if len(cases) != count:
        raise ValueError(f"Dataset must contain at least {count} cases")
    prompts = [translation_prompt(case["text"], case["target_lang"]) for case in cases]
    if len(set(prompts)) != len(prompts):
        raise ValueError("Dataset contains identical translation prompts")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Dataset contains duplicate case IDs")
    requests = [
        {**case, "id": f"r{round_index + 1:03d}-{case['id']}",
         "source_case_id": case["id"], "round_index": round_index + 1}
        for round_index in range(rounds) for case in cases
    ]
    return data, raw, requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/Hy-MT2-1.8B-NVFP4-fused.gguf")
    parser.add_argument("--binary", default="bin/hy-batch.exe")
    parser.add_argument("--dataset", default="scripts/benchmark_cases.json")
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--cache-type-k", choices=CACHE_TYPE_BYTES, default="f16")
    parser.add_argument("--cache-type-v", choices=CACHE_TYPE_BYTES, default="f16")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sampling-threads", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--seed", type=int, default=42, help="Request i uses seed + i, as in translate_many")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--cases", type=int, choices=(32, 64), default=64)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--backend-sampling", action="store_true")
    parser.add_argument("--kv-unified", action="store_true")
    parser.add_argument("--pad-final-wave", action="store_true")
    parser.add_argument("--gpu-prefix", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--refill-min", type=int, default=1)
    parser.add_argument("--disable-projections", action="store_true")
    parser.add_argument("--disable-rope", action="store_true")
    parser.add_argument("--strict-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-sparse", action="store_true")
    parser.add_argument("--disable-topk", action="store_true")
    parser.add_argument("--disable-snapshot", action="store_true")
    parser.add_argument("--disable-batch-snapshot", action="store_true")
    parser.add_argument("--eager-graphs", "--eager", action="store_true",
                        help="Capture CUDA graphs without the normal warm-up delay; does not disable CUDA graphs")
    parser.add_argument("--graph-opt", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the command/config without writing or launching")
    parser.add_argument("--compare", type=Path, help="Exact-output comparison with a prior HTTP or native JSON report")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label):
        raise ValueError("label must contain only letters, digits, dots, underscores, and hyphens")
    if not 1 <= args.parallel <= 256 or not args.parallel <= args.ubatch <= args.batch:
        raise ValueError("require 1 <= parallel <= 256 and parallel <= ubatch <= batch")
    if args.rounds < 1 or args.threads < 1 or args.timeout <= 0 or not 0 < args.max_tokens < args.context:
        raise ValueError("rounds, threads, timeout and max-tokens must be positive; max-tokens must be below context")
    if not 0 <= args.seed or args.seed + args.cases * args.rounds >= 2**32:
        raise ValueError("all request seeds must be in 0..4294967294")
    if args.parallel * args.context > 2**31 - 1:
        raise ValueError("total context must fit int32")
    if args.sampling_threads > 1 and args.disable_snapshot:
        raise ValueError("parallel sampling requires snapshots; use --sampling-threads 1 with --disable-snapshot")
    if not 1 <= args.refill_min <= args.parallel:
        raise ValueError("refill-min must be in 1..parallel")
    if sum((args.gpu_prefix, args.continuous, args.backend_sampling)) > 1:
        raise ValueError("gpu-prefix, continuous and backend-sampling are separate experimental paths")

    dataset_path = resolve(args.dataset)
    dataset, dataset_bytes, requests = build_requests(dataset_path, args.cases, args.rounds)
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    input_text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests)
    input_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    paths = {
        "input": ROOT / "results" / f"{args.label}.input.jsonl",
        "output": ROOT / "results" / f"{args.label}.jsonl",
        "summary": ROOT / "results" / f"{args.label}.summary.json",
        "config": ROOT / "results" / f"{args.label}.config.json",
        "stdout": ROOT / "results" / f"{args.label}.stdout.log",
        "stderr": ROOT / "results" / f"{args.label}.stderr.log",
        "report": ROOT / "results" / f"{args.label}.json",
    }
    env = os.environ.copy()
    env["PATH"] = str(ROOT / "runtime/cuda") + os.pathsep + str(ROOT / "runtime/msvc") + os.pathsep + env["PATH"]
    env.update(
        PYTHONUTF8="1", CUDA_CACHE_PATH=str(ROOT / "cache/cuda"),
        LLAMA_HYMT_FUSED_PROJ="0" if args.disable_projections else "1",
        GGML_CUDA_HYMT_DISABLE_ROPE_NORM=str(int(args.disable_rope)),
        GGML_CUDA_HYMT_ROPE_NORM_STRICT=str(int(args.strict_rope)),
        LLAMA_HYMT_SPARSE_PENALTIES="0" if args.disable_sparse else "1",
        GGML_CUDA_HYMT_DISABLE_TOPK=str(int(args.disable_topk)),
        LLAMA_HYMT_SAMPLING_SNAPSHOT="0" if args.disable_snapshot else "1",
        LLAMA_HYMT_BATCH_SNAPSHOT="0" if args.disable_batch_snapshot else "1",
        GGML_CUDA_HYMT_EAGER_GRAPHS=str(int(args.eager_graphs)),
        GGML_CUDA_GRAPH_OPT=str(int(args.graph_opt)),
    )
    command = [
        str(resolve(args.binary)), "--model", str(resolve(args.model)),
        "--input", str(paths["input"]), "--output", str(paths["output"]), "--summary", str(paths["summary"]),
        "--parallel", str(args.parallel), "--context", str(args.context),
        "--batch-size", str(args.batch), "--ubatch-size", str(args.ubatch),
        "--max-tokens", str(args.max_tokens), "--threads", str(args.threads), "--seed", str(args.seed),
        "--sampling-threads", str(args.sampling_threads),
    ]
    command += cache_type_args(args.cache_type_k, args.cache_type_v)
    if args.continuous:
        command += ["--refill-min", str(args.refill_min)]
    for name in ("greedy", "backend_sampling", "kv_unified", "pad_final_wave", "gpu_prefix", "continuous"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    config = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "native_command": command,
        "environment": {key: value for key, value in env.items() if key.startswith(("LLAMA_", "GGML_", "CUDA_CACHE"))},
        "dataset_path": str(dataset_path), "dataset_sha256": dataset_sha256,
        "input_jsonl_sha256": input_sha256, "requests": len(requests),
        "seed_rule": "base seed plus zero-based global request index, unchanged across wave sizes",
        "cache_type_k": args.cache_type_k, "cache_type_v": args.cache_type_v,
        "cache_cli_support_checked": False,
        "warnings": cache_type_warnings(args.cache_type_k, args.cache_type_v),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    for key in ("binary", "model"):
        if not resolve(getattr(args, key)).is_file():
            raise FileNotFoundError(resolve(getattr(args, key)))
    ensure_cache_type_support(resolve(args.binary), args.cache_type_k, args.cache_type_v, env=env, cwd=ROOT)
    config["cache_cli_support_checked"] = bool(cache_type_args(args.cache_type_k, args.cache_type_v))
    for warning in config["warnings"]:
        print(warning, file=sys.stderr)
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Choose a new label; experiment artifact already exists: {path}")
    comparison = None
    if args.compare:
        comparison = json.loads(resolve(args.compare).read_text(encoding="utf-8-sig"))
        if comparison.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Comparison dataset hash differs")
        if not args.greedy or not comparison.get("sampling", {}).get("greedy"):
            raise ValueError("Exact regression comparison requires greedy runs on both sides")

    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    paths["input"].write_bytes(input_text.encode("utf-8"))
    paths["config"].write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    process_start = time.perf_counter()
    with paths["stdout"].open("wb") as stdout, paths["stderr"].open("wb") as stderr:
        process = subprocess.run(
            command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    process_wall_s = time.perf_counter() - process_start
    if not paths["summary"].exists() or not paths["output"].exists():
        raise RuntimeError(f"Native process exited with {process.returncode}; inspect {paths['stderr']}")
    native = json.loads(paths["summary"].read_text(encoding="utf-8-sig"))
    results = [json.loads(line) for line in paths["output"].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if [item["id"] for item in results] != [item["id"] for item in requests]:
        raise ValueError("Native output IDs or request order differ from input")
    for index, (item, result) in enumerate(zip(requests, results)):
        if result.get("seed") != args.seed + index:
            raise ValueError(f"Wrong seed for {item['id']}")
        result["source_case_id"] = item["source_case_id"]
        result["round_index"] = item["round_index"]
        result["text"] = item["text"]
        if "source_lang" in item:
            result["source_lang"] = item["source_lang"]
    generated = sum(item["completion_tokens"] for item in results)
    visible = sum(item["output_tokens"] for item in results)
    if generated != native["completion_tokens"] or visible != native["output_tokens"] or len(results) != native["requests"]:
        raise ValueError("Native counters disagree with per-request JSONL")
    success = [item for item in results if item["ok"]]
    complete = [item for item in success if item["finish_reason"] == "stop"]
    errors = [item for item in results if not item["ok"]]
    truncated = [item for item in results if item["finish_reason"] == "length"]
    complete_tokens = sum(item["completion_tokens"] for item in complete)
    wall = native["wall_s"]
    if wall <= 0:
        raise ValueError("Invalid native wall-clock measurement")
    summary = {
        "wall_s": wall, "process_wall_s_including_initialization": process_wall_s,
        "initialization_s": native["initialization_s"], "successful_requests": len(success),
        "errors": len(errors), "truncated_requests": len(truncated), "complete_translations": len(complete),
        "output_tokens": generated, "output_tokens_per_second": generated / wall,
        "complete_output_tokens": complete_tokens, "complete_output_tokens_per_second": complete_tokens / wall,
        "complete_translations_per_second": len(complete) / wall,
        "text_tokens_without_eog": visible, "text_tokens_without_eog_per_second": visible / wall,
        "dummy_decode_tokens": native["dummy_decode_tokens"], "graphs_reused": native["graphs_reused"],
        "sampling_threads": native["sampling_threads"],
        "valid_throughput_run": process.returncode == 0 and not errors and not truncated,
    }
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "label": args.label,
        "engine": native["engine"], "model": str(resolve(args.model)),
        "dataset_name": dataset.get("name"), "dataset_path": str(dataset_path), "dataset_sha256": dataset_sha256,
        "input_jsonl_sha256": input_sha256, "cases": len(requests), "unique_source_cases": args.cases,
        "rounds": args.rounds, "concurrency": args.parallel, "cache_prompt": False,
        "cache_type_k": args.cache_type_k, "cache_type_v": args.cache_type_v,
        "sampling": {"temperature": 0.0 if args.greedy else 0.7, "top_p": 1.0 if args.greedy else 0.6,
                     "top_k": 1 if args.greedy else 20, "repeat_penalty": 1.05, "max_tokens": args.max_tokens,
                     "seed": args.seed, "greedy": args.greedy},
        "repeat_last_n": 4096, "min_p": 0.0,
        "warmup": {"requests": 0, "excluded_from_metrics": False},
        "exit_code": process.returncode, "summary": summary, "native_summary": native,
        "metric_notes": [
            "summary.output_tokens uses completion_tokens including terminal EOG, as the HTTP benchmark does.",
            "text_tokens_without_eog excludes terminal EOG. Both counters exclude every dummy padding token.",
            "Native wall_s includes first-use warm-up, JSONL reading/writing, templates, tokenization, prefill, sampling and decode.",
            "Native wall_s excludes model/context initialization; process_wall_s_including_initialization includes it.",
            "Historical HTTP runs used separate warm-up; benchmark_upstream.py uses no separate warm-up on either side.",
            "Requests, round IDs, prompt text and seed progression match benchmark.py and translate_many.",
            "This is a fixed smoke/performance workload, not a scored translation-quality benchmark.",
        ],
        "results": results,
    }
    if comparison is not None:
        old = {item["id"]: item for item in comparison["results"]}
        report["comparison"] = {
            "path": str(resolve(args.compare)),
            "exact_matches": sum(item["ok"] and old.get(item["id"], {}).get("ok", False)
                                 and item["translation"] == old[item["id"]]["translation"] for item in results),
            "changed_ids": [item["id"] for item in results if item["id"] in old
                            and item["translation"] != old[item["id"]]["translation"]],
            "missing_ids": [item["id"] for item in results if item["id"] not in old],
        }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(paths["report"]), **summary}, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["valid_throughput_run"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(run(parse_args()))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
