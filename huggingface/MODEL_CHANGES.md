# Model attribution and modifications

Base model: Tencent Hy-MT2-1.8B, copyright (c) 2026 Tencent.

Original model: https://huggingface.co/tencent/Hy-MT2-1.8B

Official GGUF source: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF

The original Apache-2.0 license is preserved as LICENSE in this model repository.

- Q4_K_M: repacked Tencent's official Q4_K_M GGUF without changing the source quantized tensor block bytes.
- NVFP4: quantized a BF16 GGUF from the Unsloth conversion of the same Tencent model using llama.cpp's NVFP4 tensor-type override, then repacked the result. This is reference quantization without a task-specific calibration dataset; it is not a quantization of an already low-bit model.
- Layout: joined QKV/QK and FFN gate/up projections for the project's specialized loader. Original precision differences are retained; Q4_K_M uses mixed Q4_K/Q6_K and F32 tensors.
- Repacking checks: source tensor slices were read back from the new file and compared byte-for-byte through SHA-256. This does not prove identical inference floating-point results or lossless translation quality.
- Standard-layout reconstruction: the project's unpacking script splits joined tensors without dequantizing or requantizing, verifies all 354 tensor shapes/types/byte slices, and removes only the private packed-layout metadata marker. The restored Q4_K_M file is byte-identical to Tencent's original GGUF: SHA-256 `dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699`. NVFP4 reconstruction also verifies all tensor bytes; this does not imply that NVFP4 and Q4_K_M have identical weights or quality.
- Original tokenizer/model metadata is retained. In the NVFP4 file, the top-level quantization label can still indicate Q4_0 because tensor-level overrides were used; tensor types, not that label, identify the actual NVFP4 payload.

The fused files require this project's modified inference loader and are not claimed compatible with unmodified third-party applications. Original LoRA adapters cannot be attached directly to the joined layout.

The loader, source patch, Windows runtime package and benchmark evidence are available in the [Hy-MT2-Windows project](https://github.com/divingclone/Hy-MT2-Windows). The model card's upstream Q4_K_M versus custom NVFP4 speed comparison measures the complete deployments, including different quantization, interfaces, scheduling and builds. It does not establish equivalent translation quality or attribute the entire gain to one kernel.

Exact input repositories, immutable revisions, source and output checksums, tensor counts, and packing evidence are recorded in provenance.json and manifest.json. Tencent and Unsloth do not endorse these modifications.
