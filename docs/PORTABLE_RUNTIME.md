# 免安装运行时与普通 INT4 兼容模型

0.2.2 将推理所需依赖随便携包提供。用户解压即可运行，不需手动安装 Python、Visual Studio/MSVC、Windows SDK、完整 CUDA Toolkit 或 WebView2。仍需 Windows x64 和可用的 NVIDIA 显示驱动；项目保守支持驱动 596.36 及以上，这不是 NVIDIA 官方最低驱动结论。

## 随包依赖

- PyTorch/vLLM 的 CUDA DLL、应用本地的 MSVC 可再分发运行库。
- Triton Windows 已自带的 TinyCC、PTXAS、CUDA 头文件/导入库。首次启动仍会针对具体输入形状编译 Triton 内核，缓存留在应用目录；免安装不代表没有首次编译。
- FlashInfer sampling AOT DLL，构建机用 MSVC 14.43/CUDA 13.0 生成 SM80/86/89/120f fat binary。运行时启用原来的 FlashInfer 高性能采样并设置 `FLASHINFER_DISABLE_JIT=1`，不会在用户机器构建 CUDA C++ 扩展。
- 界面优先使用系统 Evergreen WebView2。系统缺失时，自动从微软下载 Fixed Version 153.0.4234.48 x64 到 `data/webview-runtime/<版本>`，校验 CAB SHA-256 和 Microsoft 签名后解压使用，不运行安装器。下载窗口显示进度，关闭可取消，重新启动可续传。维护者仍须定期更新备用固定版本以获取安全修复。

运行环境会排除宿主机工具链 PATH、CUDA/VC/SDK/Python 缓境变量，固定使用包内工具和独立缓存。只分发允许再分发的运行组件，不把整个 Visual Studio/Windows SDK 复制进包。来源和许可保留在 `licenses/`、wheel metadata 及运行时目录中。

相关上游说明：[Triton Windows 随包编译组件与架构](https://github.com/triton-lang/triton-windows)、[MSVC 可再分发清单](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution)、[CUDA 许可](https://docs.nvidia.com/cuda/eula/)、[WebView2 Fixed Version 分发](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution)。

## 显卡路径

| 显卡 | 自动选择 | 验证状态 |
| --- | --- | --- |
| RTX 50 / SM12.0 | NVFP4 W4A4，CUTLASS | RTX 5090 实测 |
| RTX 40 / SM8.9 | 普通 GPTQ INT4 W4A16，Marlin | 运行库包含架构；尚待实卡测试 |
| RTX 30 / SM8.6 | 普通 GPTQ INT4 W4A16，Marlin | 运行库包含架构；尚待实卡测试 |
| SM8.0 Ampere | 同 INT4 路径 | 架构包含；未实卡验证 |
| GTX 16 / RTX 20 / SM7.5 | 暂不支持 | 当前 Triton 3.6 与 vLLM Windows 组合不能只换权重解决 |
| GTX 10 / SM6.1 | 暂不支持 | 当前 Triton 不支持 Pascal，需要另一套运行栈 |

按用户要求不为 GTX 10/16 维护第二套旧版依赖或恢复 llama.cpp。普通 INT4 不依赖 FP4 Tensor Core，但也不能消除底层运行库的架构要求。推荐至少 8 GB 显存；实际可用容量、上下文和并发会影响可运行配置，不保证每张同系列显卡的所有设置均可用。

所有路径默认动态 INT8 KV（包含尺度后，相同 token 池比 BF16 KV 节省 48.44%）。新增 `compat` 模式可以在 RTX 50 上显式使用 INT4。RTX 30/40 显式选择 NVFP4 时会说明应切换到自动/兼容模式，不会静默改用另一种量化。

新设置的默认显存预算比例从 30% 改为 75%，避免 8 GB 显卡在默认预算下无法启动。它是包含工作区和安全余量的配置估计上限，不是驱动硬配额；旧用户显式保存的比例保持原值。32/256 并发的默认 KV 池容量不随比例增加而自动扩大。

## 验证口径

`scripts/validate_portable_toolchain.py` 为每次运行建立空的 Triton、Inductor、vLLM、CUDA、FlashInfer 缓存；审计子进程并拒绝调用宿主机 MSVC/NVCC。非 greedy 的翻译请求覆盖 top-k/top-p 采样。FlashInfer 可能探测包内不存在的 nvcc 路径以读取版本，再回退到 `torch.version.cuda`；这不执行任何宿主机编译器。

这个检查验证依赖隔离、冷启动和功能，不把三条翻译的 TPS 当作性能数据。正式回归使用 `benchmark_portable_runtime.py`：预分词核心引擎、512 请求、32/256 并发、首轮与三轮热测分开；排除 HTTP、初始化和文件写入。INT4 保真度使用独立 256 条评价数据，以已有原始 BF16 模型输出为参考，同时收集相同 teacher 前缀下的 logits 指标。

当前实卡只有 RTX 5090。隔离本机工具链并不等同于在全新 Windows 安装或 RTX 30/40 实卡上完成测试；这些仍是交付验证边界。

### 本轮回归结果

| RTX 5090，INT8 KV | 32 并发 TPS | 256 并发 TPS |
| --- | ---: | ---: |
| NVFP4 W4A4 | 4,134.63 | 21,404.17 |
| GPTQ INT4 W4A16 | 4,162.64 | 13,251.06 |

以上为每组一次进程、三次热测的中位数，原始结果位于 `results/portable-benchmarks`。历史完整基准每组有三次独立进程；其 4,323.68 / 23,795.47 TPS 仍保留在原报告，不用本轮较小回归样本替换。性能口径记录在 `benchmarks/portable-runtime.json`。

256 条独立评价数据相对原始 BF16 teacher：INT4 的 chrF++ 为 84.9654、逐条完全一致 41/256、teacher-forced NLL 为 0.23112 nats/token、原始 logits top1 一致率 92.936%。已有 NVFP4+INT8 KV 对照分别为 79.5983、29/256、0.34843、89.341%。chrF++ 差值为 +5.367，分层配对 bootstrap 95% 区间为 [4.485, 6.267]。

这里衡量与原始模型的保真度，不以数据集译文为标准答案，也不把 chrF++ 当成翻译正确率或质量保留百分比。NVFP4 对照与 BF16 teacher 复用了之前保留的原始输出，INT4 是本轮便携运行时生成；不是严格锁定全部运行环境的纯量化算法消融。完整指标与文件哈希见 `benchmarks/int4-teacher-fidelity.json`。

Windows 10 使用固定版 WebView2 时，桌面程序会按微软文档对应用本地浏览器目录授予 AppContainer 读取权限；Windows 11 无需这一步。此目录需要是可写的本地目录，不支持把整个应用从 UNC 网络目录直接运行。

## 模型独立分发

默认程序包不含权重。[NVFP4](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-vLLM) 和 [INT4](https://huggingface.co/divingclone/Hy-MT2-1.8B-INT4-vLLM) 分别提供原始 safetensors checkpoint；模型页直接下载文件，无需 ZIP 解压或登录。按固定提交逐文件校验 SHA-256，支持续传，全部校验后原子发布目录。NVFP4 约 1.28 GiB，INT4 约 1.21 GiB。应用更新保留用户模型，ZIP 仅用于可选离线导入。
