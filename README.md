# Hy-MT2-Windows

腾讯 Hy-MT2-1.8B 的 **Windows 原生 vLLM 翻译后端**，提供本地 OpenAI 兼容 API、离线 JSONL 批量翻译及 Tauri 桌面服务管理。无需 WSL2、Docker 或云端推理。

默认按显卡选择 **RTX 50：NVFP4 / CUTLASS W4A4；RTX 30/40：GPTQ INT4 / Marlin W4A16**，均使用动态 INT8 KV。模型的共享 embedding/head 保留 BF16；BF16 源模型只用于校准和保真度评测，不是部署模型。

## 启动

下载 [0.2.3 完整便携 ZIP](https://github.com/divingclone/Hy-MT2-Windows/releases/download/desktop-v0.2.3/HyMT-0.2.3-windows-x64-portable.zip)，解压后双击 `hymt-desktop.exe`。模型在应用内单独下载；命令行入口位于 `payload/`。

完整 vLLM 便携包包含 Python、推理依赖、MSVC 运行库、Triton TinyCC/PTXAS、预编译 FlashInfer 采样内核。界面优先使用系统 WebView2，缺失时自动下载应用本地版本。用户无需手动安装 Python、MSVC、CUDA Toolkit 或 WebView2；需要已有 NVIDIA 驱动 596.36 或更新。RTX 50 已在 RTX 5090 实测；RTX 30/40 使用 INT4 兼容路径，运行库含对应架构，但尚待实卡验证。GTX 10/16、RTX 20 不在当前运行包支持范围。详见 [免安装与显卡兼容](docs/PORTABLE_RUNTIME.md)。

```powershell
.\setup-model.cmd
.\start-server.cmd -Background
# http://127.0.0.1:18080/v1，模型名 hy-mt2
.\stop-server.cmd
```

首次启动需要 JIT 编译；正常使用应保持服务常驻。服务日志位于 `results/server.stdout.log` 和 `results/server.stderr.log`。后台启动等待健康检查通过后才报告就绪，停止命令核对进程路径与创建时间并结束整个 vLLM 子进程树。

源代码环境先安装运行时：

```powershell
# 已有经验证的开发环境时可直接整理为可搬移运行时
.\setup-runtime.cmd --from-environment .local/vllm-win
# 或在具备 uv、CPython 3.12 和编译工具时安装固定依赖
.\setup-runtime.cmd
```

程序包不包含模型权重。[NVFP4](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-vLLM) 和 [INT4](https://huggingface.co/divingclone/Hy-MT2-1.8B-INT4-vLLM) 分别托管在 Hugging Face 独立仓库，根目录直接提供 `model.safetensors`、配置和 tokenizer，无需登录。首次通过模型页或 `setup-model.cmd` 按显卡下载一个模型；按固定提交逐文件校验 SHA-256，支持断点续传和已有模型复用。ZIP 仅作为可选离线导入格式。详见 [模型说明](models/README.md)。

## 批量翻译与配置

```powershell
.\translate-batch.cmd examples/input.jsonl output.jsonl --parallel 32
.\start-server.cmd -Background -Parallel 256 -ContextPerSlot 2048
# 保真优先：权重仍为 4 位，使用 Marlin 16 位激活
.\start-server.cmd -Background -Profile quality -KVCacheDtype bfloat16
```

上述服务器命令分别使用，重启前先停止旧服务。批量入口直接调用 vLLM 引擎，排除 HTTP；运行时应停止其他 GPU 推理服务。默认 32 并发、2K 上下文、2048 调度 token、总显存 75% 的估算预算。KV 是共享 token 池，不为每个请求预留完整上下文；长文本可调整上下文与 `-KVGib`。输出和旁路文件不覆盖已有文件。

输入每行：`{"id":"1","text":"Hello.","target_lang":"Chinese"}`。输出保留 ID、译文、结束状态及 token 计数。HTTP 客户端为 `scripts/translate.py`。详见 [使用说明](docs/USAGE.md) 和 [桌面端](docs/DESKTOP.md)。

0.2.3 使用 HyMT 专用 PyTorch CUDA 后端，便携包约 **966 MB**，相比 0.2.2 缩小约 **51.4%**。NVFP4/INT4 受控输出对照一致；RTX 5090 短测 NVFP4 持平、INT4 约低 2.7%，尚不能保证所有场景性能不变。构建方法、默认采样差异和验证范围见 [定制后端说明](docs/PYTORCH_CUSTOM_BUILD.md)。

## 实测依据

主对照使用**未修改的官方 llama.cpp 发布包 + 腾讯官方 Q4_K_M**，与本项目 vLLM 比较。双方使用同一 aiohttp 异步连接池客户端，通过 `POST /v1/chat/completions`、`stream: false` 翻译；计时包含请求处理、HTTP、服务端排队、推理和完整 JSON 响应解析。

2026-09-20–21，RTX 5090 / Windows 原生，固定 512 条中英双向请求；每种配置 3 个独立服务进程，每个进程先跑一轮负载，再测 3 轮。TPS 为逐级中位数，排除服务启动和结果写盘；显存为三个进程的峰值增量中位数。

| 并发 | 部署方案 | TPS（输出 token/s） | 显存峰值增量（GiB） |
| --- | --- | ---: | ---: |
| 32 | 官方 llama.cpp + Q4_K_M / F16 KV | 1,258.52 | 5.13 |
| 32 | 本项目 vLLM + NVFP4 / INT8 KV | 4,061.30 | 4.12 |
| 256 | 官方 llama.cpp + Q4_K_M / F16 KV | 1,418.02 | 8.21 |
| 256 | 本项目 vLLM + NVFP4 / INT8 KV | 18,315.16 | 5.91 |

显存每 100 ms 采样，以整段服务生命周期的设备已用显存峰值减去启动前基线，包含初始化、CUDA Graph、KV 和工作区，存在桌面噪声，不能视为精确的进程独占显存。官方程序为稳定版 v0.4.1 指定的 b10964，二进制及腾讯量化权重均校验哈希且未修改。双方上下文上限均为 2048，共享 KV token 容量相同，KV 精度如表。

优化采用异步客户端及连接池，保留单 API 前端和 2048 调度 token，模型、KV 精度和采样参数未为速度降低。结果包含量化、调度和接口差异，不代表纯内核收益或质量等效；客户端并发也不等于 GPU 实际批大小。vLLM 启动较慢，TPS 适用于服务常驻场景。完整配置、波动、延迟、启动成本和复现命令见 [API 优化报告](docs/API_OPTIMIZATION.md)，机器可读数据见 [基准摘要](benchmarks/official-nonstream-api-async.json)。

以原始 BF16 输出为参考的独立消融显示，INT8 KV 额外偏离较小，主要偏离来自权重及激活计算；文本一致性不是语义正确率，见 [BF16 参考保真度](docs/TEACHER_FIDELITY.md)。[历史同步客户端结果](docs/OFFICIAL_API_BENCHMARK.md)和[核心引擎结果](docs/VLLM_KV.md)分别保留，不与本表混算。

## 开发与历史

当前发布构建使用社区 Windows vLLM wheel 0.29.0+cu132、PyTorch 2.11.0+cu130 和本地 Hunyuan 兼容插件。固定依赖、MSVC 日志兼容补丁及构建见 [BUILD](docs/BUILD.md)。完整便携包体积较大，本轮提供便携发行流程；旧 0.1.3 安装包仍是 llama.cpp，不是此迁移结果。

旧引擎实现、GGUF 工具和原有文档归档在 [archive/llama-cpp](archive/llama-cpp/README.md)，历史源码补丁保留在 `patches/`，均不进入当前运行包。原始实验输出保留在忽略的 `results/`。模型独立发布到 Hugging Face，程序发行流程见 [RELEASE](docs/RELEASE.md)。
