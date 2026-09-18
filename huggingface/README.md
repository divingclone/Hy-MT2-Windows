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

Q4_K_M 来自腾讯官方 GGUF，经投影张量重排；NVFP4 从 BF16 GGUF 经固定 OPUS-100 中英翻译语料校准和局部加权 MSE 尺度搜索量化，再进行布局重排。NVFP4 矩阵含 4-bit 数据与分组尺度，实际为约 4.5 bit/weight，归一化等张量保留 F32。Q4_K_M 是混合量化，部分张量为 Q6_K/F32。详见 [改动与来源](MODEL_CHANGES.md)、[可核验来源记录](provenance.json) 和 [文件 SHA-256 清单](manifest.json)。

## 安装与使用

从 [GitHub Releases](https://github.com/divingclone/Hy-MT2-Windows/releases/latest) 下载 `HyMT-Windows-NVIDIA-runtime.zip` 并解压。运行包不含模型，自带 Python、CUDA/MSVC 运行库，无需另装 Python、CUDA Toolkit、Visual Studio、WSL 或 Docker。需要 Windows x64 和支持显卡的 NVIDIA 驱动 **580.88 或更新版本**。

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

默认使用 F16 K/V 缓存。需要节省显存时，批量命令增加 `--cache-type-k q8_0 --cache-type-v q8_0`，服务启动命令增加 `-CacheTypeK q8_0 -CacheTypeV q8_0`。Q8 写入融合已接入运行程序；显存与吞吐取舍应按实际长度、并发和上下文测量。

手动下载时，将 GGUF 放在环境包的 `models` 目录，保留原文件名。并发由启动时的**可用显存**自动估算，文件模式最高 256、API 最高 128，每条默认上下文 1024 token。FP16 KV cache 每条 1024 上下文约需 64 MiB，此外还要容纳模型和工作区；4 GB 显卡应从低并发开始，长文本增大上下文时应降低并发。详细手动建议见 [使用说明](https://github.com/divingclone/Hy-MT2-Windows/blob/main/docs/USAGE.md)。

运行包编译覆盖 GTX 16、RTX 20/30/40/50 对应的 `sm_75/80/86/89/120a`，没有 PTX，不能外推未列出的 GPU 架构。**只有 RTX 5090 做过实机验证**，其他型号是编译覆盖，不是速度或显存实测承诺。

## 性能与质量

当前程序与模型的 F16 / Q8 KV 速度、显存、实际逐次范围以及官方 llama.cpp 对照，统一维护在 [项目性能表](https://github.com/divingclone/Hy-MT2-Windows#实测加速) 和 [完整报告](https://github.com/divingclone/Hy-MT2-Windows/blob/main/docs/PERFORMANCE.md)。这些比较包含量化、代码、调度、接口和构建的整体差异，不能归因为单个内核，也不能把文件批处理速度当作 HTTP 服务速度。

当前 NVFP4 使用固定 OPUS-100 中英 validation 校准，独立 test 生成 256 条随机长短请求评测；两个翻译方向在每个长度组内均衡。局部校准模型的 chrF++ 为 52.346，对照未校准权重为 51.664；差值为 +0.682，配对 bootstrap 95% 区间为 [+0.264, +1.178]。完整设置和证据见 [公开语料评测](https://github.com/divingclone/Hy-MT2-Windows/blob/main/benchmarks/nvfp4-q8-mixed-evaluation.json)。这是构造子集的模型选择结果，不是完整官方 OPUS 分数或生产质量保证。

量化可能影响数值、否定、专业术语及格式。张量重排保留量化数据字节，融合计算仍可能改变浮点归约顺序；不同 K/V 类型的译文也可能不同。长文本保留完整原文和所需生成预算，按实际请求分布验证质量和延迟。重要译文应按业务要求核对。

## 许可

基座模型由 Tencent 发布，模型许可为 [Apache-2.0](LICENSE)。此仓库由社区维护，包含自行量化和布局修改，不代表腾讯官方发布或背书。修改说明见 [MODEL_CHANGES.md](MODEL_CHANGES.md)。
