"""Identical non-streaming (or explicitly streaming) HTTP translation workload.

Concurrency is the maximum number of in-flight requests, not total request count.
Use a dedicated server with prefix caching disabled. Startup is outside timing.
The first pass is reported separately; three subsequent passes form the median.
Raw results stay in the ignored results directory. No API key is written to disk.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request

from benchmark_inputs import build_requests, ROOT
from translate import translation_prompt


def percentiles(values):
    values = sorted(values)
    if not values:
        return {f"p{p}": None for p in (50, 95, 99)}
    return {f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
            for p in (50, 95, 99)}


def payload_for(case, index, backend, max_tokens, greedy=False, stream=False):
    payload = {
        "model": "hy-mt2", "messages": [{"role": "user", "content":
            translation_prompt(case["text"], case["target_lang"])}],
        "stream": stream,
        "max_tokens": max_tokens, "temperature": 0.0 if greedy else 0.7,
        "top_p": 1.0 if greedy else 0.6, "top_k": 1 if greedy else 20,
        "min_p": 0.0, "seed": 42 + index,
    }
    if stream:payload['stream_options']={'include_usage':True}
    if backend == "vllm":
        payload["repetition_penalty"] = 1.05
    else:
        payload.update(repeat_penalty=1.05, repeat_last_n=4096, cache_prompt=False)
    return payload


def read_stream(response, started):
    chunks, first, finish, usage, done = [], None, None, {}, False
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError(str(event["error"]))
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                if first is None:
                    first = time.perf_counter() - started
                chunks.append(content)
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if not done or finish is None:
        raise ValueError("Incomplete SSE stream: missing DONE or finish_reason")
    if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 1:
        raise ValueError("Missing valid server completion-token count")
    return "".join(chunks), first, finish, usage


def read_completion(response):
    return parse_completion(json.load(response))


def parse_completion(result):
    if result.get('error'):raise RuntimeError(str(result['error']))
    choices=result.get('choices',[])
    if len(choices)!=1:raise ValueError('Expected exactly one completion')
    choice=choices[0]
    content=choice.get('message',{}).get('content')
    usage=result.get('usage',{})
    if not isinstance(content,str) or choice.get('finish_reason') is None:
        raise ValueError('Incomplete non-streaming response')
    if type(usage.get('completion_tokens')) is not int or usage['completion_tokens']<1:
        raise ValueError('Missing valid server completion-token count')
    return content,None,choice['finish_reason'],usage


def one(case, index, args, submitted):
    started = time.perf_counter()
    row = {"id": case["id"], "target_lang": case["target_lang"], "ok": False,
           "translation": "", "truncated": False, "error": None,
           "client_queue_s": started - submitted, "ttft_s": None,
           "completion_tokens": 0, "seed": 42 + index}
    try:
        payload = payload_for(case, index, args.backend, args.max_tokens, args.greedy, args.stream)
        payload["model"] = args.model
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if args.stream else "application/json"}
        if key := os.environ.get("BENCHMARK_API_KEY"):
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=args.timeout) as response:
            content, first, finish, usage = read_stream(response, started) if args.stream else read_completion(response)
        row.update(translation=content, ttft_s=first, finish_reason=finish,
                   truncated=finish == "length", completion_tokens=usage["completion_tokens"],
                   prompt_tokens=usage.get("prompt_tokens"),
                   ok=bool(content.strip()) and finish == "stop")
        if not row["ok"]:
            row["error"] = "Empty, truncated, or unexpected finish reason"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["latency_s"] = time.perf_counter() - started
    row["end_to_end_s"] = row["client_queue_s"] + row["latency_s"]
    return row


def summarize(rows, wall):
    complete = [r for r in rows if r["ok"] and not r["truncated"]]
    tokens = sum(r["completion_tokens"] for r in complete)
    return {"requests": len(rows), "completed": len(complete),
            "errors": sum(not r["ok"] for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
            "wall_s": wall, "completion_tokens": tokens,
            "completion_tokens_per_second": tokens / wall,
            "requests_per_second": len(complete) / wall,
            "latency_s": percentiles([r["latency_s"] for r in complete]),
            "ttft_s": percentiles([r["ttft_s"] for r in complete if r["ttft_s"] is not None]),
            "end_to_end_s": percentiles([r["end_to_end_s"] for r in rows]),
            "valid": len(complete) == len(rows) and bool(rows)}


async def run_async_pass(cases,args):
    import aiohttp
    from http_transport import JsonClient
    trace=aiohttp.TraceConfig()
    async def sent(session,context,params):
        context.trace_request_ctx['request_headers_sent_s']=time.perf_counter()-started
    trace.on_request_headers_sent.append(sent)
    async with JsonClient(args.concurrency,args.timeout,force_close=args.no_keepalive,trace_configs=[trace]) as client:
        gate=asyncio.Semaphore(args.concurrency)
        async def worker(case,index):
            async with gate:
                begin=time.perf_counter()
                row={'id':case['id'],'target_lang':case['target_lang'],'ok':False,'translation':'',
                     'truncated':False,'error':None,'client_queue_s':begin-started,'ttft_s':None,
                     'completion_tokens':0,'seed':42+index}
                try:
                    payload=payload_for(case,index,args.backend,args.max_tokens,args.greedy,False)
                    payload['model']=args.model
                    headers={}
                    if key:=os.environ.get('BENCHMARK_API_KEY'):headers['Authorization']='Bearer '+key
                    result=await client.post(args.url.rstrip('/')+'/v1/chat/completions',payload,
                                             headers=headers,trace_request_ctx=row)
                    content,first,finish,usage=parse_completion(result)
                    row.update(translation=content,finish_reason=finish,truncated=finish=='length',
                               completion_tokens=usage['completion_tokens'],prompt_tokens=usage.get('prompt_tokens'),
                               ok=bool(content.strip()) and finish=='stop')
                    if not row['ok']:row['error']='Empty, truncated, or unexpected finish reason'
                except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
                row['latency_s']=time.perf_counter()-begin
                row['end_to_end_s']=row['client_queue_s']+row['latency_s']
                return row
        started=time.perf_counter()
        rows=await asyncio.gather(*(worker(case,index) for index,case in enumerate(cases)))
        wall=time.perf_counter()-started
    return rows,wall,client_peak(rows)


def client_peak(rows):
    events=[]
    for row in rows:events.extend(((row['client_queue_s'],1),(row['end_to_end_s'],-1)))
    active=peak=0
    for _,delta in sorted(events):active+=delta;peak=max(peak,active)
    return peak


def run_pass(cases, args):
    if getattr(args,'client','urllib')!='urllib':
        from http_transport import run_async
        return run_async(run_async_pass(cases,args),winloop=args.client=='aiohttp-winloop')
    # On Windows, creating threads while requests already run can take longer
    # than fast completions. Prestart all workers and release the complete
    # queue together. This measures offered client-task concurrency, not
    # simultaneous server arrivals: payload construction and connection work
    # can still delay other threads after the gate opens.
    ready = threading.Barrier(args.concurrency + 1, timeout=60)
    release = threading.Event()
    started = [0.0]

    def worker(case, index):
        release.wait()
        return one(case, index, args, started[0])

    with ThreadPoolExecutor(max_workers=args.concurrency, initializer=ready.wait) as pool:
        futures = [pool.submit(worker, case, index) for index, case in enumerate(cases)]
        ready.wait()
        started[0] = time.perf_counter()
        release.set()
        rows = [future.result() for future in futures]
        wall = time.perf_counter() - started[0]
    # Record client worker overlap, including preparation and response parsing.
    # This does not measure server-side active requests or GPU batch size.
    events = []
    for row in rows:
        events.extend(((row["client_queue_s"], 1), (row["end_to_end_s"], -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return rows, wall, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("llama", "vllm"), required=True)
    parser.add_argument("--url", required=True, help="Base URL without /v1")
    parser.add_argument("--model", default="hy-mt2")
    parser.add_argument("--concurrency", type=int, choices=(32, 256), required=True)
    parser.add_argument("--input", type=Path, help="Optional JSONL, otherwise 64 fixed cases x 8")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument('--stream',action='store_true',help='Opt in to SSE; default is full JSON non-streaming translation')
    parser.add_argument('--client',choices=('urllib','aiohttp','aiohttp-winloop'),
                        help='Default: pooled aiohttp for JSON, urllib for explicit SSE')
    parser.add_argument('--no-keepalive',action='store_true',help='Ablation: close async client connections after each request')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.client=args.client or ('urllib' if args.stream else 'aiohttp')
    if args.stream and args.client!='urllib':parser.error('Async transport supports non-streaming JSON only')
    if args.repeats < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("repeats, max-tokens, and timeout must be positive")
    if args.input:
        raw = args.input.read_bytes()
        cases = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    else:
        _, raw, cases = build_requests(ROOT / "scripts/benchmark_cases.json", 64, 8)
    if len(cases) < args.concurrency or len({r["id"] for r in cases}) != len(cases):
        parser.error("Input must have unique IDs and at least concurrency requests")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 3, "created_at": datetime.now(timezone.utc).isoformat(),
              'protocol':'streaming_sse' if args.stream else 'non_streaming_json',
              "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "input_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest(),
              "notes": ["First full pass is separately reported as cold; subsequent passes are warm.",
                        "Async clients use a bounded semaphore and pooled session; urllib prestarts threads before a common gate.",
                        "Timing excludes event-loop/session or thread setup and cleanup; observed_peak_in_flight counts overlapping client tasks, including payload preparation and parsing, not server arrivals or GPU batch size.",
                        "Startup/compilation excluded; cold means first request pass in this server process.",
                        "Non-streaming measures complete JSON responses; TTFT is not observable and remains null. SSE opt-in measures first text chunk.",
                        "Latency excludes client queue; end_to_end includes client queue.",
                        "Only successful stop completions count toward throughput; any failure invalidates a run.",
                        "Prefix caching must be disabled by the server; llama requests use cache_prompt=false."],
              "runs": []}
    for iteration in range(args.repeats + 1):
        rows, wall, peak = run_pass(cases, args)
        phase = "cold" if iteration == 0 else f"warm-{iteration}"
        raw_output = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        (args.output / f"{phase}.jsonl").write_bytes(raw_output)
        metrics = summarize(rows, wall)
        metrics['first_group_client_start_s']=percentiles([r['client_queue_s'] for r in rows[:args.concurrency]])
        metrics['first_group_headers_ready_s']=percentiles([r['request_headers_sent_s'] for r in rows[:args.concurrency] if 'request_headers_sent_s' in r])
        metrics["observed_peak_in_flight"] = peak
        metrics["valid"] = metrics["valid"] and peak == args.concurrency
        report["runs"].append({"phase": phase, "metrics": metrics,
                               "output_sha256": hashlib.sha256(raw_output).hexdigest()})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase": phase, **metrics}), flush=True)
        if not metrics["valid"]:
            raise RuntimeError(f"Invalid pass; see {args.output / (phase + '.jsonl')}")
    warm = [r["metrics"] for r in report["runs"][1:]]
    report["warm_median"] = {k: statistics.median(r[k] for r in warm)
        for k in ("wall_s", "completion_tokens_per_second", "requests_per_second")}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
