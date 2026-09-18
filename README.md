# Hy-MT2 Windows：NVIDIA 批量翻译加速

基于腾讯 **Hy-MT2-1.8B** 和 llama.cpp 的原生 Windows 翻译推理。提供 4 位量化、专用批量翻译程序和兼容 OpenAI 的本地 API，重点优化多条翻译的总吞吐。

项目直接修改了推理源码：合并模型投影、融合 RoPE 与归一化、稀疏重复惩罚、GPU 批量候选筛选，以及复用 CUDA Graph 的批处理调度。无需 WSL、Docker 或云端 API。见 [源码优化说明](docs/OPTIMIZATIONS.md)。

## 快速开始

1. 从本仓库 **Releases** 下载 `HyMT-Windows-NVIDIA-runtime.zip` 并解压。GitHub 的源码 ZIP 不包含运行环境。
2. 安装适合显卡的 NVIDIA 驱动 **580.88 或更新版本**。运行包自带 Python 和 CUDA/MSVC 运行库，无需另装 Python、CUDA Toolkit 或 Visual Studio。
3. 在解压目录打开终端，下载模型并翻译：

```powershell
.\setup-model.cmd
.\translate-batch.cmd examples\input.jsonl output.jsonl
```

首次下载需要联网，模型保存在 `models`，之后可离线推理。模型独立托管于 [Hugging Face](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF)，不放进 Git 仓库或运行环境 ZIP。

输入是 UTF-8 JSONL，每行一条，输出保持输入顺序和 ID：

```json
{"id":"1","text":"Please keep your ticket.","target_lang":"Chinese"}
{"id":"2","text":"请在出发前确认天气。","target_lang":"English"}
```

需要网页或 API 时运行 `start-server.cmd`，访问 <http://127.0.0.1:18080>；API 路径为 `/v1/chat/completions`，模型名 `hy-mt2`。停止服务运行 `stop-server.cmd`。

## 显卡与配置

- Windows x64，覆盖 GTX 16、RTX 20/30/40/50 对应的 CUDA 架构；**只有 RTX 5090 做过实机验证**，其余是编译覆盖。
- 自动模式在本包支持的 RTX 50 架构上使用 NVFP4，其他支持架构使用腾讯 Q4_K_M。两者都是 4 位量化，并使用本工程专用的无损权重重排布局。
- 默认根据启动时的**可用显存**选择并发：文件翻译最高 256，API 最高 128，每条上下文 1024 token。`gpu-info.cmd` 可查看检测结果。
- GTX 1650 等 4 GB 显卡可从低并发尝试，具体取决于可用显存；自动预算是估计，不是其他显卡的实测承诺。

```powershell
# 使用腾讯官方量化
.\setup-model.cmd --profile official
.\translate-batch.cmd input.jsonl output.jsonl --profile official

# 长文本：增大每条上下文，同时降低并发
.\translate-batch.cmd input.jsonl output.jsonl --parallel 32 --context 2048
```

完整硬件范围、显存建议、长文本、API 和常见问题见 [使用说明](docs/USAGE.md)。

## 实测加速

**对比未修改的原版 llama.cpp：本项目吞吐为原版的 7.46 倍。** RTX 5090 32 GB / Windows 11，512 条固定翻译；两端均为 256 并发、每条上下文 1024，分别运行三次取中位数：

| 方案 | 量化 | completion token/s | 512 条推理墙钟 |
| --- | --- | ---: | ---: |
| 官方原版 llama.cpp `b11029`，HTTP 服务 | 腾讯 Q4_K_M | 721.73 | 46.394 s |
| 本项目，专用原生批量程序 | NVFP4 | **5386.26** | **6.185 s** |

量化也是本项目优化的一部分：这组比较包含量化、推理代码、调度和接口的整体收益，**不是单一内核或纯源码的加速倍数**。原版使用未经修改的官方发布二进制；原版 Q4_K_M 文件 SHA-256 与腾讯原文件完全一致。两端推理源码基线相同，但官方包使用 CUDA 13.4 / Clang，本包使用 CUDA 13.0 / MSVC，构建差异也包含在对照中。

所有运行均完整完成 512 条，无错误、截断或空输出。计数包含真实结束 token、排除填充 token；墙钟排除模型初始化，包含首次推理与图准备。本项目含初始化的进程总时间中位数为 **7.463 秒**。两种量化的输出与质量不保证相同，5386 token/s 也不是本项目 HTTP API 的速度。

逐次范围、原版来源、计时边界、同量化附加对照及历史波动见 [性能报告](docs/PERFORMANCE.md)，机器可读证据见 [原版对比摘要](benchmarks/upstream-comparison.json)。测试只覆盖本机和固定素材，不代表所有显卡或文本分布的持续速度。

## 源码与模型

本仓库保留完整源码补丁、固定版本复现脚本，以及启动、下载和测试代码；模型、运行库、SDK、构建缓存及本机运行记录不提交 Git。基于上游提交 `bdcbaaf6e7520b68c8c60ff724c67409970d70e1`。开发者见 [源码构建](docs/BUILD.md)，维护者见 [发布流程](docs/RELEASE.md)。

`*-fused.gguf` 需要本工程修改后的加载器。重排保留原始量化权重字节，但量化本身有精度损失，计算形状变化也可能改变浮点结果。NVFP4 的速度选择不等于质量优于 Q4_K_M；对数字、否定和专业术语等内容，应按实际用途检查译文。

上游来源：[腾讯 Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)、[腾讯 GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)、[llama.cpp](https://github.com/ggml-org/llama.cpp)。代码和模型分别遵循仓库所附许可证及上游模型许可。
