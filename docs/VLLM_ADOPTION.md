# vLLM 后端迁移验收

本文保留 0.2.0 首次迁移时的验收记录。当前免安装依赖和模型下载方式见 [PORTABLE_RUNTIME](PORTABLE_RUNTIME.md)，包体积见 [RUNTIME_SIZE](RUNTIME_SIZE.md)，API 性能见 [API_OPTIMIZATION](API_OPTIMIZATION.md)。

2026-09-20，项目默认后端已切换为 Windows 原生 vLLM 0.29.0+cu132 社区构建。命令行、离线批量和桌面端共用同一运行时及配置逻辑，不经过 WSL2。

默认模型为 NVFP4 W4A4 / CUTLASS，KV 为动态 `int8_per_token_head`，并发 32，支持配置到 256。保真模式使用相同 4 位权重和 Marlin 16 位激活。BF16 源权重只保留用于校准和教师参照评测。

桌面设置兼容旧 Q8/F16 选项，显存提示改为共享 token 池。模型管理使用 SHA-256 校验的 NVFP4 ZIP；旧 GGUF 不能直接导入。进入推理运行时前规范化 Tauri 的 Windows 扩展路径，保留进程树清理、鉴权、密钥轮换和日志选项。

## 验证

- Python 回归：58 项通过。
- 前端：类型检查与生产构建通过，3 项配置测试通过。
- Rust：Windows Job Object 子进程清理测试通过。
- 真实 GPU 桌面集成：启动、普通/流式翻译、鉴权开关、密钥轮换、重新部署、无效配置保留原服务、日志开关及停止清理全部通过。
- 命令行直接引擎批量：3 条中英日样例全部成功，无截断。
- 独立便携目录：包内 Python/PyTorch/vLLM 导入、模型校验、GPU 批量翻译、HTTP 翻译及停止服务全部通过。

以上是迁移功能验收，不是新的吞吐基准。32/256 并发性能仍以 [统一核心对照](VLLM_KV.md) 为准；不能将三条样例的短批运行速度与正式预热基准混算。质量判断见 [BF16 教师参照](TEACHER_FIDELITY.md)。`benchmarks/current-*.json` 等旧文件名属于历史 llama.cpp 记录。

## 清理及产物

清理 46 个旧构建、重复权重及安装包目标，文件逻辑大小共 55.56 GiB；硬链接可能共享物理块，因此不把该数值称为实际磁盘释放量。原始测试结果保留在 `results/`，旧源码、二进制、单个基准模型及运行库归档至 `archive/llama-cpp/`，不进入新运行包。

本地便携包：`dist/vllm-0.2.0/HyMT-0.2.0-windows-x64-portable.zip`，3,953,829,021 字节，解包内容约 6.05 GiB，54,563 个文件。SHA-256：

```text
2641a67569d933c21a8ed6d75830225a922cb23e192452225160dada4477a7b4
```

包内不包含 API 密钥、测试缓存、旧 GGUF 或 llama.cpp 引擎。宿主机仍需已验证的 RTX 50 / SM 12.0、驱动 596.36 或更新、MSVC x64 C++ Build Tools、CUDA Toolkit 13.x；桌面需要 WebView2。此产物是本地便携包，尚未发布线上 Release。

本机验收清单为 `results/vllm-adoption.json`，搬迁报告为 `results/vllm-portable-validation-final.json`，逐项清理记录为 `results/vllm-adoption-cleanup.json`。

后续 0.2.1 已移除用户安装 MSVC/CUDA Toolkit/WebView2 的要求，并新增普通 INT4 兼容路径；上方 0.2.0 数据保留为迁移历史。当前要求见 [PORTABLE_RUNTIME](PORTABLE_RUNTIME.md)。
