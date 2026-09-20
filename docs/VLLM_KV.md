> 本文是迁移前的实验记录，保留原测试条件和命令。项目现已采用 vLLM；现行运行入口和默认配置见 [README](../README.md)。旧原生引擎工具已归档，不随当前运行包提供。

# Windows 原生 vLLM：8 位 KV 实测

本轮按项目默认节省显存的目标，比较 **NVFP4 权重 + 8 位 KV**，没有下载或替换 FP8 模型权重。vLLM 默认入口已改为 `int8_per_token_head`；原 BF16 KV 保留为显式对照。Windows 原生运行，未使用 WSL2。

## 主对照：vLLM INT8 与 BF16 KV

RTX 5090 32 GB；固定 512 条、32/256 并发、上下文 1024、最大生成 512。两端预分词，输入 token ID 完全一致；计时包含调度、prefill、decode、采样和文本解码，排除初始化、分词、JSON 读写和 HTTP。每种配置三个交错进程，每进程首轮 + 三次预热；主值是三个进程预热中位数的中位数。

| 并发 | BF16 KV token/s | INT8 KV token/s | 吞吐变化 | 整任务显存峰值增量：BF16 → INT8 | 显存下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 4,598.37 | 4,323.68 | -6.0% | 5.66 → 4.18 GiB | 26.1% |
| 256 | 22,469.63 | 23,795.47 | +5.9% | 9.23 → 6.27 GiB | 32.1% |

显存为每 100 ms 设备采样峰值减启动前基线，包含初始化、工作区和 CUDA Graph，存在桌面噪声，不是精确的进程独占峰值。两种 vLLM 配置采用相同可缓存 token 数，不能把缓存本身的压缩比当成整任务显存压缩比。

BF16 自动选择 FlashAttention 2；动态 INT8 使用 Triton attention。因此表中是可部署配置的整体变化，既包含 KV 精度，也包含 attention 实现变化。线性层均使用同一份 CUTLASS NVFP4 W4A4 权重，模型共享 embedding/head 仍为 BF16。

## 相对项目默认 llama.cpp Q8 的迁移收益

| 并发 | llama.cpp Q8_0 token/s | vLLM INT8 token/s | vLLM / llama.cpp | 512 条核心耗时：llama.cpp → vLLM |
| --- | ---: | ---: | ---: | ---: |
| 32 | 3,155.45 | 4,323.68 | **1.37 倍** | 10.523 → 7.619 s |
| 256 | 5,143.31 | 23,795.47 | **4.63 倍** | 6.466 → 1.381 s |

主矩阵共 **18 个进程、72 轮、每轮 512 条**，全部正常结束，无错误、空输出或截断。两端保持各自的固定波次/连续批处理方式；权重量化校准与 KV 格式不同，不能归为单个内核的加速。原生 Q8 按每槽完整上下文预留，vLLM 按总 token 池分配，两端整任务显存不可称为相同最坏情况容量下的比较。

## 缓存预算

| 并发 | BF16 缓存预算 | INT8 缓存预算 | 两者实际可缓存 token 数 |
| --- | ---: | ---: | ---: |
| 32 | 3 GiB | 1.546875 GiB | 49,152 |
| 256 | 6 GiB | 3.09375 GiB | 98,304 |

本模型 BF16 KV 每 token 为 64 KiB；vLLM 动态 INT8 为 33 KiB，包含每个 token、每个 K/V 注意力头的 FP32 缩放因子，缓存预算减少 **48.4375%**。llama.cpp Q8_0 为 34 KiB/token，属于每 32 个值带块尺度的另一种整数格式，不能与 vLLM INT8 按位等同。

256 并发的池容量并不支持 256 条同时占满 1024 token；它足够处理本次短文本负载。长文本需提高 `--kv-gib` 或降低并发后重测。日志没有抢占警告也不等于证明所有场景都零抢占。

## 独立翻译质量

本节保留此前对数据集单一参考译文的测量，不能把分数差直接归为量化劣化。新的量化保真度主对照使用原始 BF16 模型输出作为参考，并增加固定参考前缀下的 token 概率分析，见 [BF16 参考消融](TEACHER_FIDELITY.md)。两种评测回答的问题不同，分数不可混用。

独立 OPUS 中英混合子集 256 条，32 并发、上下文 4096、最大生成 2048；不截短译文。采样 temperature=0.7、top_p=0.6、top_k=20、repetition_penalty=1.05、seed=42+序号，与此前贪心质量表口径不同。

| 配置 | chrF++ | 错误 / 空输出 / 截断 |
| --- | ---: | ---: |
| llama.cpp Q8_0 | 52.0687 | 0 / 0 / 0 |
| vLLM BF16 KV | 52.0639 | 0 / 0 / 0 |
| vLLM INT8 动态 KV | 52.1214 | 0 / 0 / 0 |
| vLLM FP8 动态 KV | 52.0044 | 0 / 0 / 0 |
| vLLM FP8 E4M3，默认尺度 | 51.7730 | 0 / 0 / 0 |

INT8 相对 vLLM BF16 的差值为 +0.0575，1000 次分语言分层配对 bootstrap 的 95% 区间为 [-0.4208, +0.5208]。相对 llama.cpp Q8 为 +0.0527，区间 [-0.3920, +0.4768]。本子集未观察到显著整体下降，也没有证明跨领域质量等价或量化提高质量。未校正多重比较；分组结果与失败计数保留在数据中。

## 其他 8 位路径与 attention 补充对照

以下每项只有一个进程、三次预热，属于探索值，不混入主矩阵中位数。`bf16-triton` 保留 BF16 KV、改用 Triton；`fp8-dynamic` 为 Triton 的逐 token/head 动态 FP8；`fp8` 为 FlashInfer 的 FP8 E4M3，当前 checkpoint 没有 KV 校准尺度，使用默认尺度，不能称为校准后的 FP8 KV。

| 方案 | 并发 | 预热中位数 token/s | 显存峰值增量 |
| --- | ---: | ---: | ---: |
| bf16-triton | 256 | 21,711.29 | 9.46 GiB |
| bf16-triton | 32 | 4,591.66 | 5.63 GiB |
| fp8-dynamic | 256 | 22,062.99 | 6.31 GiB |
| fp8-dynamic | 32 | 4,144.49 | 4.27 GiB |
| fp8 | 256 | 22,352.75 | 6.98 GiB |
| fp8 | 32 | 4,569.75 | 5.00 GiB |

最初两次 FP8 尝试在 Windows 编译日志解码阶段失败，没有计入有效性能。FlashInfer 用 UTF-8 读取 MSVC/链接器的本地编码输出，可能在编译产物已生成时抛异常。`patches/flashinfer-windows-compiler-output.patch` 仅给诊断输出解码增加替换策略，不改变算子或权重；失败日志哈希与修正后结果均保留。英语诊断环境变量不足以独立解决本机问题。

## 初始化与首轮

| 配置 / 并发 | 初始化中位数（范围），秒 | 首轮核心耗时范围，秒 |
| --- | ---: | ---: |
| vLLM BF16 / 32 | 26.386（24.611–26.709） | 6.870–7.501 |
| vLLM INT8 / 32 | 27.669（26.818–28.341） | 7.800–7.920 |
| llama.cpp Q8 / 32 | 0.663（0.646–0.664） | 10.391–10.553 |
| vLLM BF16 / 256 | 29.207（28.739–30.021） | 1.564–1.637 |
| vLLM INT8 / 256 | 28.944（28.700–53.733） | 1.467–8.592 |
| llama.cpp Q8 / 256 | 0.773（0.768–0.787） | 6.327–6.491 |

保留本机磁盘编译缓存，初始化不等于全新机器安装后的启动成本。首次 JIT、图捕获和采样准备单列，常驻预热收益不能直接套到每次重新启动的短任务。

## 复现与默认配置

沿用 [迁移报告](VLLM_MIGRATION.md) 的独立 Windows 环境与 4 位模型。FlashInfer 0.6.11.post3 在本机需要一次性兼容补丁；已应用时不要重复应用：

```powershell
$compilerOutputPatch = (Resolve-Path patches/flashinfer-windows-compiler-output.patch).Path
git -C .local/vllm-win/Lib/site-packages apply --no-index $compilerOutputPatch
. scripts/vllm_windows_env.ps1
```

测试前停止其他 GPU 推理服务、保存原启动配置，结束后恢复。构建和结果目录使用新名字；源码/导入库准备见 [编译说明](BUILD.md)。矩阵脚本本身不管理生产服务。

```powershell
.venv/Scripts/python.exe scripts/build_core_benchmark.py --output .local/my-kv-core
.venv/Scripts/python.exe scripts/benchmark_vllm_kv.py --native .local/my-kv-core/hy-batch-core.exe --variants bf16 int8 llama-q8 --parallel 32 256 --trials 3 --repeats 3 --output results/my-kv-matrix
.venv/Scripts/python.exe scripts/summarize_vllm_kv.py --input results/my-kv-matrix --output results/my-kv-summary.json
```

新服务默认采用 INT8 KV，并按相同 token 容量缩小缓存预算：

```powershell
. scripts/vllm_windows_env.ps1
.local/vllm-win/Scripts/python.exe scripts/serve_vllm.py --parallel 256
# 显式复现原 BF16 KV 对照
.local/vllm-win/Scripts/python.exe scripts/serve_vllm.py --parallel 256 --kv-cache-dtype auto
```

两条命令分别启动，不要同时占用同一端口。`--kv-gib` 显式设置时优先于自动预算。`--kv-cache-dtype fp8_per_token_head` 可切换动态 FP8；主默认选择 INT8，综合本次显存、速度和质量结果，不宣称所有内核/参数的全局最优。

逐次数据、输入一致性校验、遥测、质量区间和源码补丁身份见 [公开 KV 测试数据](../benchmarks/vllm-kv-comparison.json)。原始译文和完整日志保留在本机 `results/vllm-kv/`，不随 Git 发布。
