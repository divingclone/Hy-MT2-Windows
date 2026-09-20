# 当前权重与 KV 配置

部署 checkpoint 为 Hy-MT2-1.8B-NVFP4-vllm，compressed-tensors 格式；主线性层 4 位 NVFP4，共享 embedding/head BF16。ZIP 和每个 checkpoint 文件由 `models/manifest.json` 固定大小及 SHA-256。

CUTLASS 使用 W4A4，Marlin 使用同一份 4 位权重及 16 位激活。默认 KV 为 `int8_per_token_head`，K/V 每个 token/head 有独立 FP32 动态尺度；不是 llama.cpp Q8_0。每 token 缓存 BF16 为 64 KiB，INT8 为 33 KiB。

量化脚本：`scripts/quantize_vllm_nvfp4.py`；独立环境要求见 `requirements-vllm-quant.txt`。校准与评测输入不重叠。BF16 源权重保留用于校准和 teacher 参考，日常部署不会加载它。

主要保真度结论使用 [未量化模型参考](TEACHER_FIDELITY.md)，不将数据集的单一译文当作唯一正确答案。速度和显存见 [KV 测试](VLLM_KV.md)。历史 GGUF 量化方法归档在 `archive/llama-cpp/docs/QUANTIZATION.md`。
