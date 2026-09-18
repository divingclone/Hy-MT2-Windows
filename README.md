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

RTX 5090 32 GB / Windows 11，512 条固定翻译，NVFP4、256 并发、1024 上下文。下列重复测量各运行三次，取中位数：

| 对照 | completion token/s | 比较 |
| --- | ---: | --- |
| 首轮源码优化，CPU 采样 | 3898 | 历史基线 |
| 第二轮代码，同轮 CPU 采样 | 3758 | 本轮对照 |
| 第二轮 GPU 批量前缀＋CPU 尾部采样 | **5319** | 较首轮 **+36.5%**；较同轮 CPU **+41.5%** |

5319 token/s 对应推理墙钟 **6.26 秒**；含初始化的进程总时间 **7.48 秒**。计数包含真实结束 token，排除填充 token；推理墙钟包含提示处理、采样、写出和首次图准备，排除模型初始化。这些 NVFP4 复测的 512 条输出逐字节一致。

绝对速度存在波动：后续最终构建三次中位数为 **4473 token/s**；便携构建单次验证为 **5357 token/s**，不能把单次结果当作成对加速比。另一次源码开关消融为 **1402 → 3383 token/s（2.41 倍）**，每侧仅一次且共用专用调度器。以上均为文件批处理结果，不能标作 API 速度。

完整对照、较慢的复测和测量边界见 [性能报告](docs/PERFORMANCE.md)；脱敏的逐次测量保存在 [benchmarks](benchmarks/README.md)。这些结果不代表所有显卡、语言或文本长度的速度，也不是正式翻译质量评测。

## 源码与模型

本仓库保留完整源码补丁、固定版本复现脚本，以及启动、下载和测试代码；模型、运行库、SDK、构建缓存及本机运行记录不提交 Git。基于上游提交 `bdcbaaf6e7520b68c8c60ff724c67409970d70e1`。开发者见 [源码构建](docs/BUILD.md)，维护者见 [发布流程](docs/RELEASE.md)。

`*-fused.gguf` 需要本工程修改后的加载器。重排保留原始量化权重字节，但量化本身有精度损失，计算形状变化也可能改变浮点结果。NVFP4 的速度选择不等于质量优于 Q4_K_M；对数字、否定和专业术语等内容，应按实际用途检查译文。

上游来源：[腾讯 Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)、[腾讯 GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)、[llama.cpp](https://github.com/ggml-org/llama.cpp)。代码和模型分别遵循仓库所附许可证及上游模型许可。
