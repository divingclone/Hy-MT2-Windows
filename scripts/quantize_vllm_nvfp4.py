"""Calibrate vLLM-compatible NVFP4 using disjoint local OPUS translation data.

This creates a new compressed-tensors model; the project's fused GGUF is not
rewritten. BF16 is only the quantization source. Run in the isolated native
Windows vLLM/llm-compressor environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from translate import translation_prompt


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheme", choices=("NVFP4", "NVFP4A16", "W4A16"), default="NVFP4")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--quantize-lm-head", action="store_true",
                        help="Untie and quantize output projection too; evaluate quality separately")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    calibration = [json.loads(x) for x in args.calibration.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    evaluation = [json.loads(x) for x in args.evaluation.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if {x["text"] for x in calibration} & {x["text"] for x in evaluation}:
        raise ValueError("Calibration/evaluation source-text leakage")
    if not 1 <= args.samples <= len(calibration):
        raise ValueError("Invalid calibration sample count")
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(args.source), local_files_only=True)
    texts = []
    for row in calibration[:args.samples]:
        messages = [{"role": "user", "content": translation_prompt(row["text"], row["target_lang"])},
                    {"role": "assistant", "content": row["reference"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > args.max_seq_length:
            raise ValueError("Calibration example exceeds context; increase --max-seq-length")
        texts.append({"text": text})
    model = AutoModelForCausalLM.from_pretrained(str(args.source), dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa")
    ignored = [] if args.quantize_lm_head else ["lm_head"]
    if args.quantize_lm_head:
        # Preserve source values but allow the output projection to use a
        # packed format while input embedding lookup stays BF16.
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    if args.scheme == "W4A16":
        from llmcompressor.modifiers.gptq import GPTQModifier
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignored)
    else:
        recipe = QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=ignored)
    oneshot(model=model, tokenizer=tokenizer, dataset=Dataset.from_list(texts), recipe=recipe,
            max_seq_length=args.max_seq_length, num_calibration_samples=args.samples,
            shuffle_calibration_samples=False, batch_size=1)
    model.save_pretrained(str(args.output), save_compressed=True)
    if args.quantize_lm_head:
        # vLLM's ParallelLMHead is not matched by the generic Linear target.
        # Explicitly name it without changing the calibrated tensor values.
        config_path = args.output / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for group in config["quantization_config"]["config_groups"].values():
            if "Linear" in group["targets"] and "lm_head" not in group["targets"]:
                group["targets"].append("lm_head")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tokenizer.save_pretrained(str(args.output))
    if (args.source / "chat_template.jinja").exists():
        shutil.copy2(args.source / "chat_template.jinja", args.output / "chat_template.jinja")
    evidence = {
        "scheme": args.scheme, "ignored_modules": ignored, "seed": 42,
        "quantize_lm_head": args.quantize_lm_head,
        "calibration_sha256": digest(args.calibration), "evaluation_sha256": digest(args.evaluation),
        "calibration_samples": args.samples, "max_seq_length": args.max_seq_length,
        "calibration_format": "project translation prompt plus reference assistant answer",
        "source_provenance": json.loads((args.source / "download-provenance.json").read_text()),
        "source_weights": {p.name: digest(p) for p in args.source.glob("*.safetensors")},
        "output_weights": {p.name: digest(p) for p in args.output.glob("*.safetensors")},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "llmcompressor", "compressed-tensors")},
    }
    (args.output / "quantization-provenance.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
