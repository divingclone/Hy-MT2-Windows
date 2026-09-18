# Model attribution and modifications

Base model: Tencent Hy-MT2-1.8B, copyright (c) 2026 Tencent.

Original model: https://huggingface.co/tencent/Hy-MT2-1.8B

Official GGUF source: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF

The original Apache-2.0 license is preserved as LICENSE in this model repository.

- Q4_K_M: repacked Tencent's official Q4_K_M GGUF without changing the source quantized tensor block bytes.
- NVFP4: quantized a BF16 GGUF from the Unsloth conversion of the same Tencent model using llama.cpp's NVFP4 tensor-type override, then repacked the result. This is reference quantization without a task-specific calibration dataset; it is not a quantization of an already low-bit model.
- Layout: joined QKV/QK and FFN gate/up projections for the project's specialized loader. Original precision differences are retained; Q4_K_M uses mixed Q4_K/Q6_K and F32 tensors.
- Repacking checks: source tensor slices were read back from the new file and compared byte-for-byte through SHA-256. This does not prove identical inference floating-point results or lossless translation quality.
- Original tokenizer/model metadata is retained. In the NVFP4 file, the top-level quantization label can still indicate Q4_0 because tensor-level overrides were used; tensor types, not that label, identify the actual NVFP4 payload.

The fused files require this project's modified inference loader and are not claimed compatible with unmodified third-party applications. Original LoRA adapters cannot be attached directly to the joined layout.

Exact input repositories, immutable revisions, source and output checksums, tensor counts, and packing evidence are recorded in provenance.json and manifest.json. Tencent and Unsloth do not endorse these modifications.
