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

RTX 5090 / Windows，同一非流式 API 客户端、512 条固定请求、2K 上下文；每种配置三个独立进程，各测三轮热负载，取逐级中位数。

| 并发 | 部署方案 | TPS（输出 token/s） | 显存峰值增量（GiB） |
| --- | --- | ---: | ---: |
| 32 | 官方 llama.cpp + Q4_K_M / F16 KV | 1,351.39 | 5.01 |
| 32 | 本项目 vLLM + NVFP4 / INT8 KV | 7,391.52 | 3.82 |
| 256 | 官方 llama.cpp + Q4_K_M / F16 KV | 1,500.00 | 8.16 |
| 256 | 本项目 vLLM + NVFP4 / INT8 KV | 20,462.47 | 5.89 |

官方基线为未修改的 llama.cpp b10964 与腾讯 Q4_K_M；双方 KV token 容量相同。显存按全程设备峰值减启动前基线统计，约 100 ms 采样，包含桌面波动。跑分与发布包使用同一正式运行库。详细条件见 [对比报告](docs/OFFICIAL_API_024.md)，优化与质量分析见 [32 并发优化](docs/LOW_CONCURRENCY_TPS.md)、[保真度评测](docs/TEACHER_FIDELITY.md)。

## 开发与历史

当前使用 Windows vLLM 0.29.0+cu132 和定制 PyTorch 2.11.0+cu130，便携包约 966 MB。构建与发布见 [BUILD](docs/BUILD.md)、[RELEASE](docs/RELEASE.md)，精简范围见 [定制后端](docs/PYTORCH_CUSTOM_BUILD.md)。

旧 llama.cpp 实现归档于 [archive/llama-cpp](archive/llama-cpp/README.md)。原始实验输出保留在忽略的 `results/`，公开摘要位于 [benchmarks](benchmarks/README.md)。
