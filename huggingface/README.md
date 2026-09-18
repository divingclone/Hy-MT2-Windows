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

**这些是专用 `fused` GGUF 文件，需要配合 Hy-MT2-Windows 项目修改的 llama.cpp 运行环境。不能直接假定兼容标准 llama.cpp、Ollama、LM Studio、vLLM 或 SGLang。** 本仓库提供权重；代码、源码补丁、启动脚本和运行环境位于 [GitHub 项目](https://github.com/divingclone/Hy-MT2-Windows)。

[下载 Windows 运行环境](https://github.com/divingclone/Hy-MT2-Windows/releases/latest) · [安装与显存配置](https://github.com/divingclone/Hy-MT2-Windows/blob/main/docs/USAGE.md) · [性能与完整证据](https://github.com/divingclone/Hy-MT2-Windows/blob/main/docs/PERFORMANCE.md)

| 文件 | 大小 | 推荐显卡 |
| --- | ---: | --- |
| [Hy-MT2-1.8B-NVFP4-fused.gguf](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF/resolve/main/Hy-MT2-1.8B-NVFP4-fused.gguf) | 1.01 GB | RTX 50，CC 12.0 |
| [Hy-MT2-1.8B-Q4_K_M-fused.gguf](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF/resolve/main/Hy-MT2-1.8B-Q4_K_M-fused.gguf) | 1.13 GB | GTX 16、RTX 20/30/40；RTX 50 也可用 |

Q4_K_M 来自腾讯官方 GGUF，经投影张量重排；NVFP4 从 BF16 GGUF 本地量化后重排。NVFP4 矩阵含 4-bit 数据与分组尺度，实际为约 4.5 bit/weight，归一化等张量保留 F32。Q4_K_M 是混合量化，部分张量为 Q6_K/F32。详见 [改动与来源](MODEL_CHANGES.md)、[可核验来源记录](provenance.json) 和 [文件 SHA-256 清单](manifest.json)。

## 安装与使用

从 [GitHub Releases](https://github.com/divingclone/Hy-MT2-Windows/releases/latest) 下载 `HyMT-Windows-NVIDIA-runtime.zip`（约 **515 MB**）并解压。运行包不含模型，自带 Python、CUDA/MSVC 运行库，无需另装 Python、CUDA Toolkit、Visual Studio、WSL 或 Docker。需要 Windows x64 和支持显卡的 NVIDIA 驱动 **580.88 或更新版本**。

在解压目录中执行：

```bat
setup-model.cmd
gpu-info.cmd
translate-batch.cmd examples\input.jsonl output.jsonl
```

`setup-model.cmd` 从本 Hugging Face 仓库下载模型到 `models`，校验大小和 SHA-256，支持断点续传。自动模式在支持的 RTX 50 / CC 12.0 上优先选 NVFP4，其他支持架构选 Q4_K_M；下载完成后可离线推理。也可显式执行：

```bat
setup-model.cmd --profile official
start-server.cmd -Profile official -Parallel 8
```

批量输入是 UTF-8 JSONL，每行包含 `text`、`target_lang` 和可选 `id`；输出保持输入顺序。`start-server.cmd` 提供本地网页及 OpenAI 兼容 API，地址为 `http://127.0.0.1:18080`，模型名 `hy-mt2`；停止服务使用 `stop-server.cmd`。

手动下载时，将 GGUF 放在环境包的 `models` 目录，保留原文件名。并发由启动时的**可用显存**自动估算，文件模式最高 256、API 最高 128，每条默认上下文 1024 token。FP16 KV cache 每条 1024 上下文约需 64 MiB，此外还要容纳模型和工作区；4 GB 显卡应从低并发开始，长文本增大上下文时应降低并发。详细手动建议见 [使用说明](https://github.com/divingclone/Hy-MT2-Windows/blob/main/docs/USAGE.md)。

运行包编译覆盖 GTX 16、RTX 20/30/40/50 对应的 `sm_75/80/86/89/120a`，没有 PTX，不能外推未列出的 GPU 架构。**只有 RTX 5090 做过实机验证**，其他型号是编译覆盖，不是速度或显存实测承诺。

## 性能与质量

对比**未修改的官方 llama.cpp `b11029`**。RTX 5090 32 GB / Windows 11，512 条翻译；两端均为 256 并发、每条 1024 token 上下文、batch/ubatch 2048，每侧运行三次，取中位数：

| 方案 | 量化 / 接口 | completion token/s | 512 条推理墙钟 |
| --- | --- | ---: | ---: |
| 官方原版 llama.cpp `b11029` | 腾讯 Q4_K_M / HTTP | 721.73 | 46.394 s |
| Hy-MT2-Windows 本项目 | NVFP4 / 原生批量程序 | **5386.26** | **6.185 s** |

在这组工作负载下，本项目吞吐为原版的 **7.46 倍**。这是量化选择、推理代码、调度、接口和构建共同作用下的整体收益，**不是纯源码或单一 CUDA kernel 加速倍数**。官方发布二进制未修改，基线 Q4_K_M 的整文件 SHA-256 与腾讯原始文件一致。官方包采用 CUDA 13.4 / Clang，本项目采用 CUDA 13.0 / MSVC；这些差异均在 [完整报告](https://github.com/divingclone/Hy-MT2-Windows/blob/main/docs/PERFORMANCE.md) 中记录。

六次运行均完整完成 512 条，无错误、截断或空输出。吞吐包含真实结束 token，不含填充 token；墙钟排除模型初始化，没有独立预热请求。本项目原生计时包含输入处理、采样和 JSONL 写出，原版 HTTP 计时包含请求队列但不含后续 JSONL 磁盘写出。素材为 64 条固定翻译重复 8 轮，禁用跨请求提示缓存。

这些是本机固定素材下的总吞吐结果，不是本项目 HTTP API 的速度，也不是其他显卡或文本长度的持续速度保证。两种量化的输出与质量不保证相同。

量化可能影响译文的数值、逻辑和细节。张量重排保留量化数据字节，但融合计算可能改变浮点归约顺序；没有宣称量化无损、所有输入逐字一致，也没有提供未经测量的 COMET/BLEU 分数。保持原模型支持的语言范围；重要译文应人工核对。

## 许可

基座模型由 Tencent 发布，模型许可为 [Apache-2.0](LICENSE)。此仓库由社区维护，包含自行量化和布局修改，不代表腾讯官方发布或背书。修改说明见 [MODEL_CHANGES.md](MODEL_CHANGES.md)。
