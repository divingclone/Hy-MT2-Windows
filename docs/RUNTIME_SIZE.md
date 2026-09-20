# 0.2.2 运行包精简

默认 ZIP 首轮从 0.2.1 的约 2.85 GiB 降至约 2.16 GiB；发布前再裁剪 cuDNN 可选子库，降至约 **1.99 GB（1.85 GiB）**。使用相同的普通 ZIP / Deflate 级别 5，无需分卷或额外解压软件。模型仍从 Hugging Face 的原始 checkpoint 仓库单独下载，不在此体积中。

## 后续构建：按实际执行路径裁剪

在已发布的 0.2.2 基础上，本地候选 ZIP 从 **1.988 GB 降至约 1.68 GB**。删除文件在旧 ZIP 中占 **303,856,282 字节**压缩数据、**828,181,698 字节**解压内容，共 **8,741 个文件**；ZIP 文件头和目录记录另有节省。仍使用 Deflate 级别 5，不改变加载方式或解压要求。这是本地候选构建，不代表线上 Release 已替换；最终大小和 SHA-256 见候选包旁的报告。

| 新增裁剪项 | 旧 ZIP 中压缩数据 | 解压内容 |
| --- | ---: | ---: |
| FlashAttention 2、cuRAND 主机库、NVPerf 性能分析组件 | 194,596,949 B | 465,932,376 B |
| OpenCV、torchaudio、Z3 及对应 metadata | 54,460,723 B | 146,222,381 B |
| FlashInfer AOT 已编译部分的 C/C++/CUDA 源码 | 8,720,702 B | 53,616,105 B |
| Mistral 自带的其他模型 tokenizer | 6,514,423 B | 36,925,748 B |
| Tcl/Tk、IDLE、turtledemo、ensurepip（两个 Python 前缀） | 11,992,498 B | 27,871,766 B |
| 指定依赖包的 `tests` 目录 | 4,764,298 B | 20,297,984 B |
| PyNvVideoCodec 遗留示例、PyWin32 帮助、pycountry 翻译目录 | 14,765,822 B | 28,346,714 B |
| PyTorch 的 MSVC 静态/导入链接库 | 8,040,867 B | 48,968,624 B |

最大的单项是 FA2：HyMT 的两个部署配置都显式选择 `TRITON_ATTN`，但 vLLM 的 attention 包会提前导入视觉 attention，后者又在模块级导入 FA2。`scripts/runtime_patches.py` 只把这一处导入移到实际选择 FlashAttention 的视觉模块构造分支，才得以去掉 379 MB 的未执行二进制。补丁对输入、输出源码分别校验固定 SHA-256，允许重复应用；版本或文件改变即停止构建。没有替换 kernel、伪造可用后端或改变 text attention 的实现。

对候选运行时 282 个 DLL/PYD 的 PE 依赖和加载字符串检查没有发现 cuRAND 主机 API 的使用者；实际随机采样使用已编译内核及 Triton，不依赖这个主机库。NVPerf 是 CUPTI 的可选性能指标分析组件；保留被 `torch_cpu.dll` 直接链接的 CUPTI。OpenCV 在 vLLM 中已有缺省导入分支，torchaudio 仅用于其他模型的音频处理，Z3 只服务于默认关闭的 Dynamo 调试校验。

FlashInfer 保留 AOT `sampling.dll`、Python 加载器和许可证，只裁剪特定构建目录中的 C/C++/CUDA 源文件。生产配置本来就禁用 FlashInfer JIT，AOT 加载在访问源码前完成。Triton 的 TinyCC、PTXAS、Python/CUDA/Torch 头文件及所需链接库仍完整保留。

这些规则只作用于发行包，开发环境不变。保留 NumPy/SymPy/PyTorch 的 `testing` 导入辅助模块、ISO 语言数据库和许可证，不全局删除 `.lib` 或测试辅助目录。桌面下载进度使用 Win32 API，无需 Tcl/Tk。模型权重、KV 精度、调度配置、CUTLASS/Marlin/Triton 计算代码均未修改。

首轮开发资产裁剪通过 82 项单测和 24,576 次请求的四组对照，12,288 组配对输出逐 token 相同；该性能数据仅对应首轮资产裁剪。后续依赖裁剪使用源码/二进制依赖审计、模型/API/AOT 导入检查及最小冷启动验收，不重复扩大吞吐测试矩阵。RTX 30/40 仍未做实卡验证。

打包与 `--skip-stage` 均检查固定的 vLLM/PyTorch 版本。审计已有 ZIP 的新增可裁剪文件和真实压缩字节：

```powershell
runtime/vllm/python.exe scripts/audit_runtime_size.py --archive dist/releases/desktop-v0.2.2/HyMT-0.2.2-windows-x64-portable.zip --output results/runtime-dependencies-audit.json
```

## 定制 PyTorch 后端

后续裁剪使用独立的构建和验收流程，详见 [专用 CUDA 后端构建](PYTORCH_CUSTOM_BUILD.md)。本地源码构建已将 `torch_cuda.dll` 从 **408,276,480 B 降至 160,360,448 B**，保留原 CPU/c10/Python 二进制、cuBLAS/cuBLASLt、vLLM 量化扩展和 Triton attention；关闭未使用的内置 memory-efficient attention/MAGMA，并减少默认 GPU 架构。通过 MSVC 延迟加载解除 cuFFT、cuSPARSE、cuSOLVER 的强制依赖后，候选再移除这三个库、下游 nvJitLink 和 FFTW 包装库。仅四个大库在上一版 ZIP 中就占 **498,989,102 B** 压缩数据，最终净收益还包含新 CUDA DLL 的变化。

此流程不会把新排除项直接用于官方 PyTorch。定制候选经过保留原生组件的依赖/导出审计，并保存 DLL、原 CPU/c10/Python 库和实际 GPU 架构的构建记录；打包器拒绝未完成推理验收或记录不匹配的定制运行时。最终包体积与 SHA-256 以定制输出目录的报告为准。

2026-09-21 已生成本地定制候选 `dist/pytorch-slim-release/HyMT-0.2.2-windows-x64-portable.zip`：**965,509,105 B**。相较上一轮依赖裁剪包再减少 **716,057,924 B（42.6%）**，相较已发布 0.2.2 减少 **1,022,028,867 B（51.4%）**。SHA-256 为 `d8cbf364d6bd06c5fbe19032a4d25d3da7776764584efa17f23ec9fb7a610990`。ZIP 完整性检查通过；五个目标 DLL 已移除，其他 **351 个原生文件**与上一版逐字节相同，仅替换 `torch_cuda.dll`。保留 compressed-tensors 所需的 Hadamard 变换表，不把该运行时数据误当成模型权重删除。

NVFP4/INT4 共 192 组受控配对输出逐 token 一致；短热测 NVFP4 持平，INT4 约低 2.7%，尚不能判断是否存在稳定性能回退。默认多进程采样曾出现输出差异，原始结果和限定说明均保留，详见 [本次验收记录](../benchmarks/custom-pytorch-runtime.json) 与 [构建说明](PYTORCH_CUSTOM_BUILD.md)。这是本地候选，未替换在线 Release。

## 发布前的 cuDNN 裁剪

采用 NVIDIA 官方支持的 [`GRAPH_JIT_ONLY`](https://docs.nvidia.com/deeplearning/cudnn/v1.11.0/developer/misc.html#cudnn-library-configuration) 配置。保留 `cudnn64_9.dll`、`cudnn_graph64_9.dll`、`cudnn_engines_runtime_compiled64_9.dll`，移除 `adv`、`cnn`、`ops`、`heuristic`、`engines_precompiled` 五个子库，减少 410,738,736 字节解压内容，原 ZIP 中对应压缩数据为 335,748,865 字节。启动环境强制设置该配置，避免继承宿主机的 FULL 配置。

此配置支持 Ampere 及更新架构，匹配本项目 RTX 30/40/50 范围。它不保留完整 cuDNN 的旧式卷积等接口，因此该运行包仍专用于当前 HyMT 文本翻译。项目继续使用 Triton attention、CUTLASS/Marlin 和原有 GPU 架构内核；未改动模型、KV 精度或计算代码。

2026-09-21，RTX 5090；同一组 512 请求，2K 上下文、2048 调度 token、INT8 KV。每格一个独立进程，先跑一轮再测三轮，以下为核心推理 TPS 中位数，排除 HTTP 和启动：

| 模式 | 并发 | 裁剪前 TPS | 裁剪后 TPS |
| --- | ---: | ---: | ---: |
| NVFP4 | 32 | 3,851.43 | 4,404.58 |
| NVFP4 | 256 | 23,574.81 | 23,609.69 |
| INT4 | 32 | 4,470.79 | 4,819.33 |
| INT4 | 256 | 16,131.92 | 16,106.43 |

32 轮共 16,384 请求全部成功，8,192 组配对结果的译文、token ID、token 数和结束状态全部一致。NVFP4、INT4 另外通过独立空缓存、非 greedy 采样和宿主编译工具隔离检查。未发现可重复的性能回退；单进程样本不用于宣称裁剪带来加速，也不替换 README 的多进程 API 对照数据。数据见 [裁剪回归摘要](../benchmarks/cudnn-runtime-pruning.json)。

复现时准备裁剪前后两个独立便携目录，使用相同模型：

```powershell
runtime/vllm/python.exe scripts/benchmark_runtime_pruning.py --baseline-root dist/before --candidate-root dist/after --model-root . --baseline-cudnn-config FULL --output results/pruning-recheck
```

比较两个已裁剪 cuDNN 的版本时省略 `--baseline-cudnn-config FULL`，两端都使用生产环境的 `GRAPH_JIT_ONLY`。可用 `--repeats 5` 增加热测轮数，或用 `--candidate-first` 反转运行顺序做独立复核。

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

保留实际推理内核及 vLLM 原有 SM80/86/89/120f 架构。对于官方 `torch_cuda.dll`，cuFFT、cuSPARSE、cuSOLVER 仍是直接依赖，不能按功能名称直接删除；上方专用构建通过重编链接关系解除这一限制。早期保留的 FA2、OpenCV 和音频组件已继续裁剪，torchvision 等仍被上游启动导入的组件继续保留。定制构建不替换本项目使用的 Triton/CUTLASS/Marlin 推理实现。

## 验证结果

RTX 5090、默认 INT8 KV，512 请求，每组一个独立进程和三轮热测；预分词核心吞吐，排除 HTTP、初始化和文件写入：

| 模式 | 精简前 32 / 256 TPS | 精简后 32 / 256 TPS |
| --- | ---: | ---: |
| NVFP4 | 4,135 / 21,404 | 4,467 / 19,693 |
| 普通 INT4 | 4,163 / 13,251 | 4,351 / 14,516 |

NVFP4 256 并发单轮只有约 1–2 秒，波动明显，因此额外做了同机原运行时与精简运行时各九轮热测对照：中位数分别为 **21,297 / 23,416 TPS**。本轮未观察到可重复的性能下降，也不把较高数值归因为裁剪带来的加速；这些小规模回归不替代历史多进程完整基准。

四组各 512 条译文及输出 token 数均与精简前保存结果逐条一致。权重、KV 精度和计算代码未改动。本轮检验裁剪回归，不重新定义翻译质量评分。运行时冷启动另外使用空缓存，拒绝调用宿主机 MSVC/NVCC，并实际执行非 greedy 采样。RTX 30/40 的保留架构尚待实卡验证。

机器可读数据见 `benchmarks/runtime-slimming.json`。生成发布 ZIP 后，再用 `scripts/validate_fixed_webview.py --mode system` 验证系统浏览器，以及 `--mode fallback` 强制覆盖缺失时的下载分支；不会卸载或修改系统 WebView2。测试产物应写在 ZIP 生成之后，避免用户配置或模型进入分发包。
