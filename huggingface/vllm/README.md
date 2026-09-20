---
license: apache-2.0
base_model: tencent/Hy-MT2-1.8B
pipeline_tag: translation
tags:
- vllm
- compressed-tensors
- quantized
---

# Hy-MT2-1.8B {VARIANT} for vLLM

A calibrated 4-bit checkpoint of Tencent Hy-MT2-1.8B for the
[Hy-MT2 Windows project](https://github.com/divingclone/Hy-MT2-Windows).
Weights, configuration, tokenizer and chat template are stored directly at this
repository root. No ZIP extraction is needed. Downloads are public and require no login.

Two model repositories are available:

- [NVFP4 W4A4](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-vLLM):
  RTX 50 fast profile, CUTLASS, 1,373,024,141 checkpoint bytes.
- [GPTQ INT4 W4A16](https://huggingface.co/divingclone/Hy-MT2-1.8B-INT4-vLLM):
  symmetric group size 128, RTX 30/40 compatibility profile, Marlin;
  also usable on RTX 50, 1,300,648,141 checkpoint bytes.

Both retain BF16 tied embedding/output-head weights. The project's default KV cache
is INT8 per token/head, a separate runtime setting. These are compressed-tensors
safetensors checkpoints, not GGUF files.

## Use

The Windows app downloads the selected repository's individual files from a pinned
commit and checks each file's size and SHA-256. Choose one model for your GPU.
CLI users run `setup-model.cmd` or `setup-model.cmd --profile compat`.
Alternatively, use the official Hugging Face client:

```python
from huggingface_hub import snapshot_download
model_path = snapshot_download("{REPO}")
# Load model_path with the project's pinned native Windows vLLM runtime/plugin.
```

Validated runtime: community SystemPanic/vllm-windows 0.29.0, PyTorch 2.11.0+cu130,
with the project's native Hunyuan model plugin. A standard repository layout does
not guarantee compatibility with arbitrary upstream vLLM/Transformers versions.
RTX 5090 was tested. RTX 30/40 support is based on runtime/kernel architecture
support and still requires hardware validation. The current packaged runtime
does not support GTX 10/16 or RTX 20.

## Provenance and fidelity

Base revision: `tencent/Hy-MT2-1.8B` at
`9a341cd1b679d3efd23b46e847b01745a71ed792`. Calibration used 256 OPUS translation
examples, disjoint from the 256-example evaluation set. File hashes are recorded
in `checksums.json`. The INT4 repository also contains `quantization.json` with
versions, calibration hashes and source/output weight hashes.

Using unquantized BF16 output as the teacher, the project measured chrF++ 84.97
for INT4/INT8 KV and 79.60 for NVFP4/INT8 KV. These are teacher-output similarity
scores, not percentages of translation quality retained; multiple translations
may be valid. INT4 evaluation reused earlier BF16/NVFP4 results, so it is not an
isolated quantizer-only experiment.

The original Apache-2.0 model license is included as `LICENSE`. See
`MODEL_CHANGES.md` for modifications. This is an independent quantization project,
not an official Tencent release.
