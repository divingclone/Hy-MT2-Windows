"""Check benchmark validity and backend sampling parity without a GPU."""
import io
import json
from pathlib import Path
import sys
import time
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_backend import payload_for, read_stream, read_completion, summarize, run_pass


class BenchmarkTests(unittest.TestCase):
    def test_workers_start_from_common_gate_and_reach_requested_concurrency(self):
        rendezvous = threading.Barrier(3, timeout=5)
        submissions = []

        def fake_request(case, index, args, submitted):
            start = time.perf_counter()
            submissions.append(submitted)
            rendezvous.wait()
            end = time.perf_counter()
            return {"client_queue_s": start - submitted,
                    "latency_s": end - start, "end_to_end_s": end - submitted}

        with patch("benchmark_backend.one", fake_request):
            rows, wall, peak = run_pass([{}] * 6, SimpleNamespace(concurrency=3))
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(set(submissions)), 1)
        self.assertEqual(peak, 3)
        self.assertGreaterEqual(wall, max(row["end_to_end_s"] for row in rows))

    def test_penalty_is_not_silently_ignored_by_vllm(self):
        case = {"text": "Hello.", "target_lang": "Chinese"}
        llama = payload_for(case, 3, "llama", 512)
        vllm = payload_for(case, 3, "vllm", 512)
        self.assertEqual(llama["repeat_penalty"], vllm["repetition_penalty"])
        self.assertNotIn("repeat_penalty", vllm)
        self.assertEqual(llama["messages"], vllm["messages"])
        self.assertEqual(vllm["seed"], 45)
        self.assertFalse(vllm['stream'])
        self.assertNotIn('stream_options',vllm)

    def test_full_json_response_requires_usage_and_finish(self):
        result={'choices':[{'message':{'content':'你好'},'finish_reason':'stop'}],
                'usage':{'completion_tokens':2,'prompt_tokens':12}}
        content,first,finish,usage=read_completion(io.BytesIO(json.dumps(result).encode()))
        self.assertEqual((content,first,finish,usage['completion_tokens']),('你好',None,'stop',2))
        result['usage']={}
        with self.assertRaises(ValueError):read_completion(io.BytesIO(json.dumps(result).encode()))

    def test_stream_requires_token_usage_and_completion(self):
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "你好"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 12}},
        ]
        wire = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        content, first, finish, usage = read_stream(io.BytesIO(wire + b"data: [DONE]\n"), time.perf_counter())
        self.assertEqual(content, "你好")
        self.assertIsNotNone(first)
        self.assertEqual(usage["completion_tokens"], 3)
        with self.assertRaises(ValueError):
            read_stream(io.BytesIO(wire), time.perf_counter())

    def test_failures_invalidate_pass_and_do_not_inflate_throughput(self):
        rows = [{"ok": True, "truncated": False, "completion_tokens": 10,
                 "latency_s": 1, "ttft_s": .1, "end_to_end_s": 1},
                {"ok": False, "truncated": True, "completion_tokens": 512,
                 "latency_s": 2, "ttft_s": .2, "end_to_end_s": 3}]
        result = summarize(rows, 2)
        self.assertFalse(result["valid"])
        self.assertEqual(result["completion_tokens_per_second"], 5)
        self.assertEqual(result["errors"], 1)


if __name__ == "__main__":
    unittest.main()
