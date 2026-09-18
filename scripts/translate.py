#!/usr/bin/env python3
"""Standard-library client for the local Hy-MT2 llama.cpp server.

Examples:
  python scripts/translate.py "今天天气真好。" --target English
  python scripts/translate.py --input-jsonl input.jsonl --concurrency 32 --output output.jsonl

JSONL records: {"id":"1", "text":"Hello.", "target_lang":"Chinese"}.
The server must load the model's GGUF chat template. No system prompt is added.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request


LANGUAGES = {
    "en": "English", "zh": "Chinese", "zh-cn": "Chinese",
    "zh-hans": "Chinese", "zh-tw": "Traditional Chinese",
    "zh-hant": "Traditional Chinese", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian",
    "pt": "Portuguese", "it": "Italian", "ar": "Arabic",
    "英语": "English", "中文": "Chinese", "汉语": "Chinese",
    "日语": "Japanese", "韩语": "Korean", "法语": "French", "德语": "German",
}


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.6
    top_k: int = 20
    repeat_penalty: float = 1.05
    max_tokens: int = 512
    seed: int = 42
    greedy: bool = False


def normalize_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    if not url.startswith(("http://", "https://")):
        raise ValueError("Server URL must begin with http:// or https://")
    return url


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    # This is a local inference client: avoid sending localhost traffic through OS proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise ValueError("Server returned a JSON value that is not an object")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result


def resolve_model(url: str, model: str | None, timeout: float) -> str:
    if model:
        return model
    models = request_json(normalize_url(url) + "/v1/models", None, timeout).get("data", [])
    if not models or not models[0].get("id"):
        raise RuntimeError("No model found at /v1/models; specify --model or start the server")
    return str(models[0]["id"])


def translation_prompt(text: str, target_lang: str) -> str:
    if not isinstance(target_lang, str):
        raise ValueError("target_lang must be a string")
    target = LANGUAGES.get(target_lang.strip().lower(), target_lang.strip())
    if not target:
        raise ValueError("target_lang must not be empty")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    return (
        f"Translate the following text into {target}. Note that you should only output "
        f"the translated result without any additional explanation:\n{text}"
    )


def translate_one(
    case: dict[str, Any], *, url: str, model: str, sampling: Sampling,
    timeout: float = 300, cache_prompt: bool = False, submitted_at: float | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    result: dict[str, Any] = {
        "id": case.get("id"), "text": case.get("text"),
        "source_lang": case.get("source_lang"),
        "target_lang": case.get("target_lang", case.get("target", "English")),
        "seed": sampling.seed, "ok": False, "translation": "", "error": None,
        "finish_reason": None, "truncated": False, "completion_tokens": None,
        "prompt_tokens": None, "timings": {},
        "client_queue_s": max(0.0, start - submitted_at) if submitted_at is not None else 0.0,
    }
    try:
        prompt = translation_prompt(case["text"], result["target_lang"])
        payload = {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "max_tokens": sampling.max_tokens,
            "temperature": 0.0 if sampling.greedy else sampling.temperature,
            "top_p": 1.0 if sampling.greedy else sampling.top_p,
            "top_k": 1 if sampling.greedy else sampling.top_k,
            "min_p": 0.0, "repeat_penalty": sampling.repeat_penalty,
            "repeat_last_n": 4096, "seed": sampling.seed,
            "cache_prompt": cache_prompt,
        }
        response = request_json(normalize_url(url) + "/v1/chat/completions", payload, timeout)
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("Server response has no choices")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Server response message.content is not a string")
        usage = response.get("usage") or {}
        timings = response.get("timings") or {}
        result.update({
            "translation": content, "finish_reason": choice.get("finish_reason"),
            "truncated": choice.get("finish_reason") == "length",
            "completion_tokens": usage.get("completion_tokens", timings.get("predicted_n")),
            "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n")),
            "timings": timings, "usage": usage, "response_id": response.get("id"),
            "ok": bool(content.strip()),
        })
        if not result["ok"]:
            result["error"] = "Server returned an empty translation"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["latency_s"] = time.perf_counter() - start
    result["end_to_end_latency_s"] = result["client_queue_s"] + result["latency_s"]
    return result


def translate_many(
    cases: list[dict[str, Any]], *, url: str, model: str, sampling: Sampling,
    concurrency: int, timeout: float = 300, cache_prompt: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    results: list[dict[str, Any] | None] = [None] * len(cases)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="hy-mt") as executor:
        futures = {}
        for index, case in enumerate(cases):
            settings = Sampling(**{**asdict(sampling), "seed": sampling.seed + index})
            future = executor.submit(
                translate_one, case, url=url, model=model, sampling=settings,
                timeout=timeout, cache_prompt=cache_prompt, submitted_at=time.perf_counter(),
            )
            futures[future] = index
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    elapsed = time.perf_counter() - start
    return [result for result in results if result is not None], elapsed


def add_client_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default="http://127.0.0.1:18080", help="Server base URL")
    parser.add_argument("--model", help="Model alias; default: discover /v1/models")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Request i uses seed + i")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding for regression comparison")


def sampling_from_args(args: argparse.Namespace) -> Sampling:
    if args.max_tokens <= 0 or args.concurrency <= 0 or args.timeout <= 0:
        raise ValueError("max-tokens, concurrency and timeout must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < 0 or args.repeat_penalty <= 0:
        raise ValueError("Invalid sampling parameters")
    return Sampling(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repeat_penalty=args.repeat_penalty, max_tokens=args.max_tokens,
        seed=args.seed, greedy=args.greedy,
    )


def load_jsonl(path: Path, default_target: str) -> list[dict[str, Any]]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict) or not isinstance(case.get("text"), str):
            raise ValueError(f"{path}:{number}: expected an object containing string text")
        case.setdefault("id", str(number))
        case.setdefault("target_lang", case.get("target", default_target))
        translation_prompt(case["text"], case["target_lang"])
        cases.append(case)
    if not cases:
        raise ValueError("Input JSONL has no records")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--text-file", type=Path, help="UTF-8 text file for one translation")
    parser.add_argument("--input-jsonl", type=Path, help="Batch JSONL: id, text, target_lang")
    parser.add_argument("--target", default="English", help="Full language name or common language code")
    parser.add_argument("--output", type=Path, help="Single result JSON or batch results JSONL")
    parser.add_argument("--json", action="store_true", help="Print full result metadata for a single translation")
    parser.add_argument("--cache-prompt", action="store_true", help="Allow server prompt cache reuse")
    add_client_arguments(parser)
    args = parser.parse_args()
    try:
        if sum(value is not None for value in (args.text, args.text_file, args.input_jsonl)) != 1:
            parser.error("Provide exactly one of text, --text-file or --input-jsonl")
        sampling = sampling_from_args(args)
        if args.input_jsonl:
            cases = load_jsonl(args.input_jsonl, args.target)
        else:
            text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig")
            cases = [{"id": "1", "text": text, "target_lang": args.target}]
            translation_prompt(text, args.target)
        model = resolve_model(args.url, args.model, args.timeout)
        results, elapsed = translate_many(
            cases, url=args.url, model=model, sampling=sampling, concurrency=args.concurrency,
            timeout=args.timeout, cache_prompt=args.cache_prompt,
        )
        serialized = "\n".join(json.dumps(result, ensure_ascii=False) for result in results) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        elif args.input_jsonl or args.json:
            sys.stdout.write(serialized)
        else:
            print(results[0]["translation"])
        errors = sum(not result["ok"] for result in results)
        truncated = sum(result["truncated"] for result in results)
        tokens = sum(result["completion_tokens"] or 0 for result in results)
        print(
            f"{len(results)} requests, {errors} errors, {truncated} truncated, "
            f"{tokens} output tokens, {elapsed:.3f} s, {tokens / elapsed:.2f} output tokens/s",
            file=sys.stderr,
        )
        for result in results:
            if result["error"]:
                print(f"{result['id']}: {result['error']}", file=sys.stderr)
        return 1 if errors or truncated else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
