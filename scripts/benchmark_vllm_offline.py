"""Measure native Windows vLLM batch inference with the project's exact prompts.

Engine initialization, compilation and graph capture are reported separately.
All 512 jobs are queued; max_num_seqs controls active sequence concurrency.
This is a batch-path comparison, separate from benchmark_backend.py's HTTP path.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

from benchmark_inputs import build_requests, ROOT
from translate import translation_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="LLM keyword arguments JSON")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--teacher-output", help="JSONL with teacher token_ids, or 'self' for this run's last pass; collect teacher-forced raw prompt logprobs")
    parser.add_argument("--pretokenized", action="store_true",
                        help="Core benchmark: prepare/tokenize once, exclude JSON output assembly from timing")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config.get("enable_prefix_caching", True):
        raise ValueError("Disable prefix caching for this repeated-prompt benchmark")
    if args.input:
        raw = args.input.read_bytes()
        cases = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    else:
        _, raw, cases = build_requests(ROOT / "scripts/benchmark_cases.json", 64, 8)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # FlashInfer invokes ninja by name; a venv's Scripts folder is not on PATH
    # when its Python is called directly rather than through activation.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM, SamplingParams
    import torch
    report = {"config": config, "environment": {k: os.environ.get(k) for k in
              ("HYMT_VLLM_NATIVE", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING")},
              "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "requests": len(cases), "sampling": {"temperature": 0 if args.greedy else .7,
              "top_p": 1 if args.greedy else .6, "top_k": 1 if args.greedy else 20,
              "repetition_penalty": 1.05, "max_tokens": args.max_tokens, "seed": "42 + index"},
              "versions": {p: importlib.metadata.version(p) for p in
                ("vllm", "torch", "transformers", "compressed-tensors", "triton-windows")},
              "notes": ["Engine init includes compilation and CUDA graph capture; excluded from inference wall.",
                        "Prompt rendering, tokenization, submission, generation and result assembly are timed.",
                        "Output JSONL writing is outside inference timing.",
                        "No padding tokens count; terminal EOS is counted only when present in returned IDs."],
              "runs": []}
    start = time.perf_counter()
    llm = LLM(**config)
    report["initialization_s"] = time.perf_counter() - start
    tokenizer = llm.get_tokenizer()
    core_prompts = None
    if args.pretokenized:
        prep_started = time.perf_counter()
        core_prompts = [{"prompt_token_ids": tokenizer.encode(tokenizer.apply_chat_template(
            [{"role": "user", "content": translation_prompt(row["text"], row["target_lang"])}],
            tokenize=False, add_generation_prompt=True), add_special_tokens=True)} for row in cases]
        report["preprocess_s"] = time.perf_counter() - prep_started
        report["metric_profile"] = "pretokenized_engine"
        report["notes"] = ["Core wall includes sampling parameter setup, submission, scheduling, prefill, decode, sampling and detokenization.",
                           "Model initialization, input parsing/tokenization and output JSON assembly/writing are excluded.",
                           "No prefix cache; first complete pass and subsequent warm passes are separate."]
        (args.output / "prompt_ids.json").write_text(json.dumps([
            {"id": row["id"], "token_ids": prompt["prompt_token_ids"]}
            for row, prompt in zip(cases, core_prompts)]) + "\n", encoding="utf-8")
    for repeat in range(args.repeats + 1):
        phase_start_unix_s = time.time()
        started = time.perf_counter()
        prompts = core_prompts if core_prompts is not None else [tokenizer.apply_chat_template([{"role": "user", "content":
            translation_prompt(row["text"], row["target_lang"])}], tokenize=False,
            add_generation_prompt=True) for row in cases]
        params = [SamplingParams(temperature=0 if args.greedy else .7,
            top_p=1 if args.greedy else .6, top_k=1 if args.greedy else 20,
            repetition_penalty=1.05, max_tokens=args.max_tokens, seed=42+i)
            for i in range(len(cases))]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if args.pretokenized:
            torch.cuda.synchronize()
            core_wall = time.perf_counter() - started
        rows = []
        if len(outputs) != len(cases):
            raise RuntimeError("Output count differs from input count")
        for case, output in zip(cases, outputs):
            candidate = output.outputs[0]
            rows.append({"id": case["id"], "target_lang": case["target_lang"],
                         "translation": candidate.text,
                         "ok": bool(candidate.text.strip()) and candidate.finish_reason == "stop",
                         "truncated": candidate.finish_reason == "length",
                         "finish_reason": candidate.finish_reason,
                         "completion_tokens": len(candidate.token_ids),
                         "terminal_eos_present": bool(candidate.token_ids) and candidate.token_ids[-1] == tokenizer.eos_token_id,
                         "prompt_tokens": len(output.prompt_token_ids),
                         "metrics": asdict(output.metrics) if is_dataclass(output.metrics) else None})
            if args.save_token_ids or args.teacher_output:
                rows[-1]["token_ids"] = list(candidate.token_ids)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        if args.pretokenized:
            wall = core_wall
        tokens = sum(row["completion_tokens"] for row in rows if row["ok"])
        phase = "cold" if repeat == 0 else f"warm-{repeat}"
        metrics = {"phase": phase, "wall_s": wall, "completion_tokens": tokens,
                   "phase_start_unix_s": phase_start_unix_s,
                   "phase_end_unix_s": phase_start_unix_s + wall,
                   "completion_tokens_per_second": tokens / wall,
                   "requests_per_second": sum(row["ok"] for row in rows) / wall,
                   "errors": sum(not row["ok"] for row in rows),
                   "truncated": sum(row["truncated"] for row in rows),
                   "terminal_eos_count": sum(row["terminal_eos_present"] for row in rows),
                   "valid": all(row["ok"] for row in rows)}
        output_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (args.output / f"{phase}.jsonl").write_bytes(output_bytes)
        metrics["output_sha256"] = hashlib.sha256(output_bytes).hexdigest()
        report["runs"].append(metrics)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics), flush=True)
        if not metrics["valid"]:
            raise RuntimeError("Invalid throughput pass; inspect saved output")
    report["warm_median"] = {key: statistics.median(row[key] for row in report["runs"][1:])
        for key in ("wall_s", "completion_tokens_per_second", "requests_per_second")}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.teacher_output:
        if core_prompts is None:
            raise ValueError("Teacher forcing requires --pretokenized")
        teacher = rows if args.teacher_output == "self" else [json.loads(line) for line in
            Path(args.teacher_output).read_text(encoding="utf-8").splitlines() if line.strip()]
        if [r["id"] for r in teacher] != [r["id"] for r in cases]:
            raise ValueError("Teacher IDs/order differ from input")
        forced_prompts = []
        for prompt, row in zip(core_prompts, teacher):
            if not row["ok"] or row["truncated"] or not row.get("token_ids"):
                raise ValueError("Teacher must have valid complete token IDs")
            ids = prompt["prompt_token_ids"] + row["token_ids"]
            if len(ids) + 1 > config["max_model_len"]:
                raise ValueError("Teacher sequence exceeds context")
            forced_prompts.append({"prompt_token_ids": ids})
        # Each reference token is scored against the identical reference prefix.
        # Raw prompt logprobs exclude generation penalties; never mix this timing
        # with free-generation throughput. Triton decoder attention reads KV cache
        # for prefill too, so quantized KV participates in this calculation.
        forced = llm.generate(forced_prompts, SamplingParams(temperature=0,
            max_tokens=1, prompt_logprobs=1, detokenize=False), use_tqdm=False)
        with (args.output / "teacher-forced.jsonl").open("w", encoding="utf-8") as stream:
            for case, prompt, ref, output in zip(cases, core_prompts, teacher, forced):
                offset = len(prompt["prompt_token_ids"])
                if len(output.prompt_logprobs) != len(output.prompt_token_ids):
                    raise RuntimeError("Incomplete prompt logprobs")
                scores = []
                for pos, token in enumerate(ref["token_ids"], offset):
                    distribution = output.prompt_logprobs[pos]
                    lp = distribution[token]
                    top = max(distribution, key=lambda t: distribution[t].logprob)
                    scores.append({"token": token, "logprob": lp.logprob,
                                   "rank": lp.rank, "top1": top})
                stream.write(json.dumps({"id": case["id"], "scores": scores}) + "\n")
        report["teacher_forcing"] = {"reference": args.teacher_output,
            "tokens": sum(len(row["token_ids"]) for row in teacher),
            "mode": "raw prompt logprobs, fixed teacher prefix, includes terminal EOS; separate prefill diagnostic"}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
