# Hy-MT2-Windows

腾讯 Hy-MT2-1.8B 的 **Windows 原生 vLLM 翻译后端**，提供 OpenAI 兼容 API、JSONL 批量翻译和 Tauri 桌面管理，无需 WSL2 或 Docker。

## 启动

下载 [0.2.4 完整便携 ZIP](https://github.com/divingclone/Hy-MT2-Windows/releases/download/desktop-v0.2.4/HyMT-0.2.4-windows-x64-portable.zip)，解压后运行 `hymt-desktop.exe`，在模型页下载模型并启动服务。运行环境已随包提供，WebView2 缺失时自动下载。

需要 NVIDIA RTX 30/40/50 和 596.36 或更新驱动。RTX 50 使用 NVFP4 / CUTLASS，RTX 30/40 使用 INT4 / Marlin，默认 INT8 KV；目前仅 RTX 5090 实卡验证。详见 [兼容说明](docs/PORTABLE_RUNTIME.md)。

命令行入口位于便携包的 `payload/`：

```powershell
.\setup-model.cmd
.\start-server.cmd -Background
# http://127.0.0.1:18080/v1，模型名 hy-mt2
.\stop-server.cmd
```

源码环境复用同版便携包的 `payload/runtime/`。首次启动需要编译，日常使用建议保持服务常驻。模型独立托管于 Hugging Face，程序包不含权重，详见 [模型说明](models/README.md)。

## 批量翻译与配置

```powershell
.\translate-batch.cmd examples/input.jsonl output.jsonl --parallel 32
.\start-server.cmd -Background -Parallel 256 -ContextPerSlot 2048
```

默认 32 并发、2K 上下文、2048 调度 token，按总显存 75% 估算预算。KV 为共享池；修改配置后重启服务。输入格式、参数和桌面操作见 [使用说明](docs/USAGE.md) 与 [桌面端](docs/DESKTOP.md)。

## 实测依据

RTX 5090，服务与客户端均为 32 并发；0.2.4 将 CUDA Graph 覆盖从 64 提高到 256 token，模型精度和采样参数不变。

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 输出 TPS | 4,758 | **7,700（+61.8%）** |
| P95 响应时间 | 552 ms | **330 ms** |
| 请求阶段显存峰值 | 3,796 MiB | **3,900 MiB（+104 MiB）** |

TPS 为三组独立进程复测的中位数；显存另测两组、约 100 ms 采样。跑分与便携包使用同一正式运行库；首次重新编译时，新旧启动峰值均约 4.02 GiB。详细条件与限制见 [优化报告](docs/LOW_CONCURRENCY_TPS.md)；[历史官方对照](docs/API_OPTIMIZATION.md)和[保真度评测](docs/TEACHER_FIDELITY.md)单独保留。

## 开发与历史

当前使用 Windows vLLM 0.29.0+cu132 和定制 PyTorch 2.11.0+cu130，便携包约 966 MB。构建与发布见 [BUILD](docs/BUILD.md)、[RELEASE](docs/RELEASE.md)，精简范围见 [定制后端](docs/PYTORCH_CUSTOM_BUILD.md)。

旧 llama.cpp 实现归档于 [archive/llama-cpp](archive/llama-cpp/README.md)。原始实验输出保留在忽略的 `results/`，公开摘要位于 [benchmarks](benchmarks/README.md)。
