---
license: apache-2.0
base_model: tencent/Hy-MT2-1.8B
pipeline_tag: translation
tags:
  - gguf
  - nvfp4
  - q4-k-m
  - windows
  - cuda
  - hy-mt2
---

# Hy-MT2-1.8B · NVFP4 / Q4_K_M · Windows NVIDIA

腾讯 [Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) 的两种量化模型，面向 Windows + NVIDIA GPU 的本地批量翻译。

**这些是专用 `fused` GGUF 文件，需要配合 Hy-MT Windows 项目修改的 llama.cpp 运行环境。不能直接假定兼容标准 llama.cpp、Ollama、LM Studio、vLLM 或 SGLang。** 本仓库提供权重；完整运行环境、一键启动脚本和配置说明随项目的 Windows 发布包提供。

| 文件 | 大小 | 推荐显卡 |
| --- | ---: | --- |
| [Hy-MT2-1.8B-NVFP4-fused.gguf](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF/resolve/main/Hy-MT2-1.8B-NVFP4-fused.gguf) | 1.01 GB | RTX 50，CC 12.0 |
| [Hy-MT2-1.8B-Q4_K_M-fused.gguf](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF/resolve/main/Hy-MT2-1.8B-Q4_K_M-fused.gguf) | 1.13 GB | GTX 16、RTX 20/30/40；RTX 50 也可用 |

Q4_K_M 来自腾讯官方 GGUF，经投影张量重排；NVFP4 从 BF16 GGUF 本地量化后重排。NVFP4 矩阵含 4-bit 数据与分组尺度，实际为约 4.5 bit/weight，归一化等张量保留 F32。Q4_K_M 是混合量化，部分张量为 Q6_K/F32。详见 [改动与来源](MODEL_CHANGES.md)、[可核验来源记录](provenance.json) 和 [文件 SHA-256 清单](manifest.json)。

## 使用

在项目 Windows 环境包目录中执行：

```bat
setup-model.cmd
gpu-info.cmd
translate-batch.cmd examples\input.jsonl output.jsonl
```

`setup-model.cmd` 按显卡自动下载合适的模型，并检查完整 SHA-256。也可显式执行：

```bat
setup-model.cmd --profile official
start-server.cmd -Profile official -Parallel 8
```

手动下载时，将 GGUF 放在环境包的 `models` 目录，保留原文件名。并发由启动时可用显存自动估算；模型文件大小不等于完整推理显存，KV cache 随上下文与并发增加。4 GB 显卡应从低并发开始；其他显卡架构仅编译覆盖，当前仅 RTX 5090 实机验证。

## 性能与质量

RTX 5090、Windows 11、NVFP4、256 并发、每条 1024 token 上下文：专用批量程序对 512 条翻译的三次测试中位数为 **5319 completion token/s**，同阶段 CPU 采样路径为 **3758 token/s**，提升约 **41.5%**。这里的“CPU 采样”仍然使用 GPU 模型推理。通用发布构建另一次验证为 **5357 token/s**，推理阶段约 **6.22 秒**，含初始化进程总耗时约 **7.64 秒**。

吞吐计入真实结束 token，不计填充 token；推理墙钟包含提示处理、采样和输出写出，不包含模型初始化。素材为 64 条中英请求重复 8 轮，禁用跨请求提示缓存。后续复测出现波动；这些结果不是其他显卡的速度承诺，也不是与原版框架的通用加速比。

量化可能影响译文的数值、逻辑和细节。张量重排保留量化数据字节，但融合计算可能改变浮点归约顺序；没有宣称量化无损、所有输入逐字一致，也没有提供未经测量的 COMET/BLEU 分数。保持原模型支持的语言范围；重要译文应人工核对。

## 许可

基座模型由 Tencent 发布，模型许可为 [Apache-2.0](LICENSE)。此仓库由社区维护，包含自行量化和布局修改，不代表腾讯官方发布或背书。修改说明见 [MODEL_CHANGES.md](MODEL_CHANGES.md)。
