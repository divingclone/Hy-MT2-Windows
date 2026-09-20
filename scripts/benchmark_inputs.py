import json
from pathlib import Path
from translate import translation_prompt
ROOT=Path(__file__).resolve().parents[1]

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
