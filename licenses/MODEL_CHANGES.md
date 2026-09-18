# Model attribution and modifications

Original model: Tencent Hy-MT2-1.8B, copyright (C) 2026 Tencent.

Official source: https://huggingface.co/tencent/Hy-MT2-1.8B

Official GGUF source: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF

The included license was retrieved from official model revision
`9a341cd1b679d3efd23b46e847b01745a71ed792` and identifies Apache License 2.0.
The original complete text is preserved in `model-Hy-MT2.txt`.

This package contains modified model files, not Tencent's unmodified download:

- `Hy-MT2-1.8B-Q4_K_M-fused.gguf` repacks the official Q4_K_M GGUF into joined
  QKV/QK and gate/up projection tensors used by this modified inference build.
  The repacking preserves the source quantized tensor block bytes.
- `Hy-MT2-1.8B-NVFP4-fused.gguf` was quantized locally from the model's BF16 GGUF
  to the NVFP4 format and similarly repacked for joined projections.
- Quantization and tensor layout changes were performed in this project.
  Tencent does not endorse this package or guarantee its modifications.

The SHA-256 of each distributed model file is recorded in
`MANIFEST.sha256.json` and `SHA256SUMS.txt`. These hashes identify the packaged
artifacts; they do not establish translation quality or numerical equivalence.
