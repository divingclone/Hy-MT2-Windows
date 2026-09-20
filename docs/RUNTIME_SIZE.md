# 0.2.2 运行包精简

默认 ZIP 从 0.2.1 的约 2.85 GiB 降至约 2.16 GiB，减少约 702 MiB（24%）；解压文件减少约 1.56 GiB。模型仍从 Hugging Face 的原始 checkpoint 仓库单独下载，不在此体积中。

## WebView2

优先探测系统 Evergreen WebView2，即使本地存在固定版本，也优先使用系统版本。缺失时自动下载经固定 SHA-256 和 Microsoft Authenticode 签名校验的固定版，保存到程序旁的 `data/webview-runtime/<版本>`；不运行安装器，不需要管理员权限。首次缺失时需联网，原生进度窗口支持关闭取消，重新启动续传。中断解压的临时目录在下次启动时清理，成功后移除下载 CAB。

因此默认包省去约 298 MiB 压缩数据。缺少系统运行时的机器仍需额外下载约 294 MiB 的 CAB，并占用约 667 MiB 解压空间；这部分是按需提供，并未消失。维护者须持续更新备用固定版本以获得安全修复。离线包用 `--include-webview2` 显式加入固定版。

## 其他精简

下表是旧 ZIP 中移除文件的压缩大小，含四舍五入：

| 内容 | 减少 |
| --- | ---: |
| NVIDIA Python 开发组件中重复的运行库、NVVM 与链接库 | 139 MiB |
| 多 GPU cuSOLVER Mg | 72 MiB |
| 未使用的备用 NVRTC DLL | 36 MiB |
| 可选 MoE 内核与重复 Qutlass 扩展 | 65 MiB |
| Numba / llvmlite 可选推测解码依赖 | 39 MiB |
| TileLang 可选后端 | 29 MiB |
| TorchCodec / PyNvVideoCodec 视频组件 | 24 MiB |

规则统一在 `scripts/runtime_filter.py`，同时用于 staging 和 `--skip-stage` 的最终打包，源开发运行时保持完整。仅针对本项目固定版本、HyMT 稠密文本模型、Triton attention 和现有采样路径验证；不作为任意模型的通用 vLLM 发行版。版本变化时构建会要求重新验证裁剪规则。

保留所有实际推理内核及原有 SM80/86/89/120f 架构。cuFFT、cuSPARSE、cuSOLVER 等是 `torch_cuda.dll` 的直接依赖，不能按功能名称直接删除。FlashAttention、OpenCV、音频/图像相关的一部分包仍被上游启动导入，因此保留。进一步大幅缩小 PyTorch/CUDA 需要维护定制构建，本轮没有修改其计算实现。

## 验证结果

RTX 5090、默认 INT8 KV，512 请求，每组一个独立进程和三轮热测；预分词核心吞吐，排除 HTTP、初始化和文件写入：

| 模式 | 精简前 32 / 256 TPS | 精简后 32 / 256 TPS |
| --- | ---: | ---: |
| NVFP4 | 4,135 / 21,404 | 4,467 / 19,693 |
| 普通 INT4 | 4,163 / 13,251 | 4,351 / 14,516 |

NVFP4 256 并发单轮只有约 1–2 秒，波动明显，因此额外做了同机原运行时与精简运行时各九轮热测对照：中位数分别为 **21,297 / 23,416 TPS**。本轮未观察到可重复的性能下降，也不把较高数值归因为裁剪带来的加速；这些小规模回归不替代历史多进程完整基准。

四组各 512 条译文及输出 token 数均与精简前保存结果逐条一致。权重、KV 精度和计算代码未改动。本轮检验裁剪回归，不重新定义翻译质量评分。运行时冷启动另外使用空缓存，拒绝调用宿主机 MSVC/NVCC，并实际执行非 greedy 采样。RTX 30/40 的保留架构尚待实卡验证。

机器可读数据见 `benchmarks/runtime-slimming.json`。生成发布 ZIP 后，再用 `scripts/validate_fixed_webview.py --mode system` 验证系统浏览器，以及 `--mode fallback` 强制覆盖缺失时的下载分支；不会卸载或修改系统 WebView2。测试产物应写在 ZIP 生成之后，避免用户配置或模型进入分发包。
