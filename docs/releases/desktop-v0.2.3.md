# HyMT 0.2.3 · 定制 PyTorch 精简运行时

Windows 便携包约 **966 MB**，相比 0.2.2 的 1.988 GB 减少约 **51.4%**。解压整个 ZIP 后运行 `hymt-desktop.exe`；已有用户可通过应用内更新。模型继续单独下载，已有模型和设置可复用。

## 本次变化

- 基于固定 PyTorch 2.11.0+cu130 源码，仅重建 CUDA 后端，将 `torch_cuda.dll` 从 408 MB 缩小至 160 MB；保留原 CPU/c10/Python 二进制。
- 延迟加载未用于 HyMT 文本推理的 cuFFT、cuSPARSE、cuSOLVER，并移除这三个 DLL、FFTW 包装库及下游 nvJitLink。
- 移除未使用的 FlashAttention 2、OpenCV、音频组件、分析工具和开发资产；将 vLLM 视觉模块的可选 FlashAttention 导入延后。
- 模型权重、INT8 KV、生产调度配置及实际使用的 Triton/CUTLASS/Marlin 推理实现保持不变。保留冷启动所需 TinyCC、PTXAS、头文件和许可文件。

## 验证范围与限制

RTX 5090 上，NVFP4 和 INT4 各使用 32 条输入、一次冷启动与两次热测；确定性离线对照共 384 次请求完成，192 组配对输出逐 token 一致。候选没有从宿主 CUDA 目录补载已删除的库。

短热测 NVFP4 吞吐持平，INT4 约低 2.7%；样本较短，尚不能判断是否有稳定回退，也不能保证所有场景性能完全不变。此前默认多进程 NVFP4 采样最后一轮有 19/32 条输出不同，所有请求正常结束；受控调度下未出现该差异，原始记录和限制已保留。产品未切换为验证使用的确定性调度。

RTX 30/40 保留相应架构和 INT4 路径，尚未实卡验证。此运行时专用于 HyMT 文本翻译，不是完整功能的通用 PyTorch 分发版；缺失 FFT/稀疏矩阵/求解器依赖的操作会明确报错。

## 下载文件

- `HyMT-0.2.3-windows-x64-portable.zip`：完整便携程序，不含模型权重。
- `SHA256SUMS.txt`：附件 SHA-256 校验值。
- `desktop-latest.json`、`.zip.sig`：应用内更新描述与 Tauri 签名。
- `pruning-report.json`：实际包体积、构建身份与验证摘要。

需要受支持的 NVIDIA RTX 显卡和 596.36 或更新驱动。WebView2 优先使用系统版本，缺失时自动下载并校验。附件使用已有 Tauri 更新签名，不含 Windows Authenticode 代码签名。

构建与裁剪说明见仓库 `docs/PYTORCH_CUSTOM_BUILD.md`、`docs/RUNTIME_SIZE.md`；性能限制见 `benchmarks/custom-pytorch-runtime.json`。
