# Model attribution and modifications

Original model: Tencent Hy-MT2-1.8B, copyright (C) 2026 Tencent.
Source: https://huggingface.co/tencent/Hy-MT2-1.8B at revision
`9a341cd1b679d3efd23b46e847b01745a71ed792`.
The original Apache-2.0 license text is preserved in `licenses/model-Hy-MT2.txt`.

The project distributes two modified checkpoint variants in separate repositories:

- NVFP4 W4A4, compressed-tensors format, group size 16. Linear weights and input
  activations use four-bit floating-point quantization; activation scaling is
  dynamic locally with calibrated global scales. Tied embedding/output weights
  remain BF16.
- GPTQ INT4 W4A16, compressed-tensors format, symmetric weight quantization with
  group size 128. Tied embedding/output weights remain BF16. Quantization versions,
  source hashes and output hashes are recorded in `benchmarks/int4-quantization.json`.

Both were calibrated on 256 OPUS translation examples, disjoint from the project's
teacher-output evaluation examples. Configuration was normalized for the pinned
vLLM runtime and the project's native Hunyuan model plugin. Tokenizer and chat
template accompany each checkpoint. These are independent quantizations of the
source safetensors checkpoint, not conversions from the previous GGUF artifacts.

Every checkpoint file's size and SHA-256 are recorded in `models/manifest.json`. KV cache quantization is selected at
runtime and does not modify these checkpoint files. Tencent does not endorse
this packaging or guarantee its modifications.
