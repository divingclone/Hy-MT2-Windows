> 本文是迁移前的实验记录，保留原测试条件和命令。项目现已采用 vLLM；现行运行入口和默认配置见 [README](../README.md)。旧原生引擎工具已归档，不随当前运行包提供。

# Windows 原生 vLLM 迁移实测

本轮评估对象是现有 Hy-MT2-1.8B 优化版 llama.cpp 与 Windows 原生 vLLM，均使用 **4 位权重**。没有使用 WSL2、Docker 或远程 GPU。桌面端和默认启动脚本仍使用原来的 llama.cpp；vLLM 是独立、可复现的实验后端。

本页保留原 F16/BF16 KV 对照。按项目默认 8 位 KV 的后续测试见 [8 位 KV 报告](VLLM_KV.md)：vLLM INT8 相对 llama.cpp Q8 的核心吞吐为 1.37 / 4.63 倍；新 vLLM 服务入口默认使用 INT8，复现本页 BF16 表格需显式指定 `--kv-cache-dtype auto`。

## 主对照：推理引擎核心吞吐

评估后端性能应以核心推理为主，HTTP 作为额外的服务开销单列。此前用 HTTP 的 2.71 / 5.95 倍作为主结论不合适，也不能将历史原生批量的 5,426.65 token/s 与 HTTP 的 784.64 token/s 直接比较。

为此补测了统一计时边界的原生引擎对照：两端均提前准备相同的 512 条提示 token ID，关闭前缀缓存，计时包含请求/采样器设置、KV 清理与调度、prefill、decode、采样和文本解码；排除模型初始化、输入解析、提示构造/tokenization、输出 JSON 组装/写盘及 HTTP。这是完整推理引擎吞吐，尚未单独隔离 GPU kernel 时间。

每个后端、每种并发各启动 3 次进程，交错执行；每个进程先跑 1 次完整首轮，再跑 3 次预热轮。主表取三个进程各自预热中位数的中位数，避免用一个后端首轮与另一个后端热态比较。

| 并发 | llama.cpp NVFP4 / F16 KV | vLLM NVFP4 CUTLASS | vLLM / llama.cpp | 吞吐提升 | 512 条核心耗时：llama.cpp → vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 3,728.57 token/s | 4,640.71 token/s | **1.24 倍** | **+24.5%** | 8.937 → 7.140 s |
| 256 | 5,840.89 token/s | 22,421.32 token/s | **3.84 倍** | **+283.9%** | 5.704 → 1.475 s |

三个进程的预热中位数范围：32 并发，llama.cpp 为 3,687–3,840、vLLM 为 4,597–4,755 token/s；256 并发分别为 5,671–6,044 和 21,697–22,527 token/s。两端采样、量化产生不同输出长度，所以请求耗时比与 token/s 比不会完全相同。

48 轮全部通过，每轮 512 条，无错误、空输出或截断。全部 12 个进程的输入 token ID 逐条一致，共 41,768 个提示 token；llama.cpp 插桩后的全部 24 轮输出与未插桩程序的对应并发输出逐条一致。插桩程序在独立目录编译，复用现有推理 DLL，没有改动生产二进制。

llama.cpp 保留项目固定波次调度，vLLM 保留连续批处理；调度是引擎性能的一部分。两端主干均为 4 位，但量化校准、embedding 精度和计算路径存在差异，因此这是两套实际引擎配置的比较，不能解释为相同权重数值下的纯内核加速。速度优先候选的留出质量仍需参考下文：CUTLASS chrF++ 为 52.0639，现有方案为 52.5523，不能宣称无损。

### 初始化与首轮单列

以下为三个进程的中位数与最小–最大范围，单位秒；首轮是 512 条的核心耗时，排除初始化。本机保留磁盘编译缓存，不能视为全新机器冷启动。

| 后端 / 并发 | 引擎初始化：中位数（范围） | 首轮核心耗时：中位数（范围） |
| --- | ---: | ---: |
| llama.cpp / 32 | 0.723（0.680–0.746） | 8.875（8.595–9.061） |
| vLLM / 32 | 28.961（28.780–125.440） | 7.346（7.250–13.800） |
| llama.cpp / 256 | 0.871（0.866–0.935） | 5.546（5.507–6.143） |
| vLLM / 256 | 32.710（29.816–51.480） | 1.643（1.569–7.932） |

初始化不是完整进程启动时间；vLLM 初始化包含模型编译与图捕获，首轮还可能产生采样 JIT。常驻服务应关注预热吞吐，短批任务则必须计入启动成本。首轮中位数不能掩盖第一次进程的较大开销，因此同时保留范围和逐次结果。

### 历史 5,426 token/s 的复核

历史 5,426.65 使用原生 `hy-batch` 完整批量墙钟，包含提示处理、首次图准备及输出写盘；它与主表剔除提示处理/写盘并单列首次轮的预热核心指标不同。此前复测的 4,340.81 也属于完整批量首轮，不应充当本表热态分母。

另外找回历史完全相同 SHA-256 的可执行文件，与当前原版可执行文件使用相同推理 DLL，按原完整批量口径同期交错各测三次：历史程序中位数 **4,946.51**，当前程序 **5,017.14 token/s**，当前约高 1.4%。没有复现此前约 20% 的旧版到当前版退化。两者与历史 5,426.65 仍有差距，历史记录缺少频率和负载证据，无法确定当时差异的具体原因。

核心对照及历史程序复核的逐次指标、输出哈希、构建身份、有效性校验和每 100 ms GPU 遥测摘要见 [核心对照数据](../benchmarks/vllm-core-comparison.json)。未锁定 GPU 频率；遥测用于保留条件，不用于事后推定历史波动原因。原生 `llama_perf` 把批量 decode 也计入 `p_eval`，不能用它推算独立的 prefill/decode GPU 耗时。

## 补充：HTTP 端到端性能

全部使用同一预先启动线程、统一放行的 HTTP 客户端。表内为 3 次预热轮中位数，每轮 512 条；两端实际在途峰值均达到 32/256，全部成功，无空译文、截断或请求错误。

| 方案 | 32 并发 token/s | 512 条耗时 | 256 并发 token/s | 512 条耗时 |
| --- | ---: | ---: | ---: | ---: |
| 现有 llama.cpp（默认 API 采样） | 1,383.28 | 24.069 s | 784.64 | 42.409 s |
| vLLM CUTLASS，W4A4 | 3,749.04 | 8.827 s | 4,671.84 | 7.074 s |
| vLLM Marlin，W4A16 | 3,685.39 | 9.015 s | 4,406.42 | 7.538 s |

CUTLASS 对当前默认 HTTP 后端的吞吐为 **2.71 倍 / 5.95 倍**（32 / 256）。这是端到端整体迁移收益，包含模型量化、计算、调度、Python/HTTP 与客户端开销，不是纯 GPU kernel 倍数。

补充检查了现有 llama.cpp 的 `--backend-sampling`：各做首次轮与 **1 次预热探索轮**，32/256 并发预热分别为 836.61/506.93 token/s，比默认采样慢，未采用。它们单独保存在聚合数据的 `exploratory=true` 记录中，没有与主表三次重复混合。

| 方案 | 32 并发 TTFT P95 | 32 并发请求耗时 P95 | 256 并发 TTFT P95 | 256 并发请求耗时 P95 |
| --- | ---: | ---: | ---: | ---: |
| 现有 llama.cpp（默认 API 采样） | 0.373 s | 1.870 s | 10.634 s | 26.951 s |
| vLLM CUTLASS，W4A4 | 0.391 s | 0.779 s | 4.727 s | 5.675 s |
| vLLM Marlin，W4A16 | 0.422 s | 0.825 s | 4.356 s | 5.867 s |

延迟列为每轮 P95 的中位数，不包含客户端等待空闲并发槽的时间；完整 512 条耗时包含该队列。32 并发下 CUTLASS 的首段延迟没有优于现有后端，但整条请求完成更快。高并发改善了吞吐和请求完成时间，仍不能用离线 23k token/s 代表 HTTP 实际吞吐。

主表使用默认流式设置（vLLM `stream_interval=1`）。另测 `stream_interval=8`，32/256 预热吞吐分别为 3,667.79/4,132.60 token/s，没有测到提升，因此不采用。这个选项改变流式片段粒度，未混入默认流式主表。

首次完整 HTTP 轮（已有本机编译缓存、排除服务启动）的 512 条耗时：

- 现有 llama.cpp（默认 API 采样）：32 并发 24.032 s，256 并发 35.687 s。
- vLLM CUTLASS，W4A4：32 并发 9.364 s，256 并发 7.246 s。
- vLLM Marlin，W4A16：32 并发 9.414 s，256 并发 13.683 s。

首次轮不是全新机器首次安装/首次编译的延迟；缓存未命中时会增加 JIT 时间。API 启动耗时、显存、每轮 token 数和输出哈希保存在[公开聚合结果](../benchmarks/vllm-windows-migration.json)。

## 早期批量探索与参数调优

以下保留调参阶段的 completion token/s。vLLM 每项是同一进程首次完整轮 + 三次预热轮；llama.cpp 是三次新进程的首轮中位数，包含首次图准备。这组计时没有统一剔除提示处理，且冷热状态不同，**只作为探索记录，不用于计算主对照的迁移加速比**；统一口径结果以上方核心主表为准。

| 方案 | 32 并发首次轮 | 32 并发预热中位数 | 256 并发首次轮 | 256 并发预热中位数 |
| --- | ---: | ---: | ---: | ---: |
| 现有 llama.cpp，F16 KV | 3,223 | 未测 | 4,341 | 未测 |
| 现有 llama.cpp，Q8 KV | 2,666 | 未测 | 4,224 | 未测 |
| vLLM CUTLASS，W4A4 | 2,639 | **4,983** | 4,325 | **23,151** |
| vLLM Marlin，W4A16 | 2,384 | 4,664 | 3,804 | 14,575 |

表内 4,341 与历史 5,427 的差异已补做历史可执行文件同期 A/B，见上文；不能把这次探索的较低值用作主对照分母。HTTP 的 5.95 倍也不代表核心引擎加速。

vLLM CUTLASS 的 512 条预热墙钟分别为 6.65 s 和 1.43 s，首次轮为 12.56 s 和 7.65 s；这还不包含引擎初始化的 27.85 s / 47.12 s。首次请求中的采样 JIT 是显著成本，反复新开进程的单批任务不一定受益。已有编译缓存也不代表没有首次推理延迟。

| vLLM 调整（全部保持 4 位主干） | 32 并发预热 token/s | 256 并发预热 token/s | 选择 |
| --- | ---: | ---: | --- |
| CUTLASS，预填充预算 2048，默认融合 | **4,983** | **23,151** | 速度优先配置 |
| lm_head 也使用 NVFP4 | 4,374 | 22,425 | 本次没有吞吐收益，不采用 |
| 额外 norm/quant 与 QK/RoPE 融合 | 4,328 | 21,432 | 本次更慢，不采用 |
| 额外融合 + 8192 预填充预算 | 4,015 | 22,524 | 中位数没有提升，不采用 |
| Marlin，16 位激活 | 4,664 | 14,575 | 质量更接近现有版本的备选 |

这是测试组合中的最优结果，不是所有参数的全局最优证明。32 并发存在明显运行波动，CUTLASS 三轮为 4,468–5,316 token/s；不能只选最快一轮。没有采用额外融合和输出层量化的候选，因此没有把它们当成已经通过质量验证的部署配置。

设备峰值显存增量估计：llama.cpp F16 在 32/256 并发约 3.72/17.73 GiB，Q8 约 2.92/10.54 GiB；vLLM CUTLASS 约 5.66/9.10 GiB。vLLM KV 预算为 3/6 GiB，llama.cpp 按每槽完整上下文预留，因此这不是相同最坏情况容量下的显存比。

## 留出翻译质量

使用独立 256 条 OPUS-100 中英混合请求，128 单句、64 中等长度、64 长文本，两种目标语言各 128 条；以 32 并发、上下文 4096、生成预算 2048 评测，没有为速度截短请求。全部候选 256/256 成功，没有空译文或截断。评分是这个固定子集上的 chrF++，不是官方完整 OPUS-100 结果；没有证明所有并发和文本长度下质量等价。

| 候选 | chrF++ | 对现有模型差值 | 配对 bootstrap 95% 区间 |
| --- | ---: | ---: | --- |
| 现有 llama.cpp NVFP4 / F16 KV | 52.5523 | — | — |
| vLLM CUTLASS NVFP4 W4A4 | 52.0639 | −0.4883 | [−1.0151, +0.0498] |
| vLLM Marlin W4A16 | 52.6356 | +0.0833 | [−0.3917, +0.5401] |

1000 次配对重采样，按目标语言分层。W4A4 英文子集下降 0.5919，区间 [−1.1877, −0.0138]；分组比较未经多重比较校正，但足以提醒不能宣称质量无损。W4A16 的分数更接近现有版本，也不能据此证明所有领域质量等价。这些差异同时包含量化、内核数值与采样器差异。

## 条件与边界

- 2026-09-20，Windows 11，单张 RTX 5090 32 GB，驱动 596.36；MSVC 14.43、CUDA Toolkit 13.0。
- vLLM 使用 [SystemPanic 0.29.0 Windows 社区构建](https://github.com/SystemPanic/vllm-windows/releases/tag/v0.29.0)，CPython 3.12、PyTorch 2.11.0+cu130、Triton Windows 3.6.0.post26。[官方安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)没有原生 Windows 官方支持承诺。
- 固定性能集：仓库 `scripts/benchmark_cases.json` 的 64 条中英请求重复 8 次，共 512 条；分别限制 32 和 256 个在途请求。上下文 1024，最大生成 512，temperature=0.7、top_p=0.6、top_k=20、repetition_penalty=1.05、seed=42+序号。
- 两端关闭前缀缓存。HTTP 同用 `scripts/benchmark_backend.py`，所有客户端线程预先就绪、512 个任务入队后统一放行；记录实际在途峰值，未达到指定并发的轮次无效。先报告首次完整请求轮，再测 3 轮预热后中位数。模型初始化与客户端线程创建/销毁不计入推理时间。TTFT 是首个非空 SSE 文本片段的到达时间。
- 核心主对照两端均预分词、同进程预热，首轮与初始化另列；llama.cpp 使用项目的固定波次 `hy-batch`，vLLM 使用连续批处理。早期批量探索与 HTTP 表各自保留原计时边界，不与主表混算。
- token/s 计实际生成 token，含真实 EOS，不含填充 token；错误、截断或空译文使整轮无效。不同采样器、量化、调度产生不同译文，因此同时报告请求数、总耗时与质量。
- vLLM 的 BF16 dtype 指计算/未量化部分的类型，模型主干仍是打包的 NVFP4 权重，KV 使用 BF16。CUTLASS 使用 W4A4；Marlin 使用同一份 4 位权重、16 位激活（W4A16）。两者都不是 FP8 模型；4 位权重不代表 4 位 KV。

早期 HTTP 客户端边创建线程边发送请求，在 Windows 上未能对快速后端充分施加 256 并发。该版 schema 1 的结果全部排除，最终 HTTP 对照只使用修正后的 schema 2。完整失败与弃用尝试保留在本机 `results/vllm-migration/`，不把它们拼接进最终中位数。

## 为什么下载 BF16

项目现有的是专用融合布局的 GGUF。此次 vLLM 路径使用 compressed-tensors checkpoint，不能直接复用该专用文件。BF16 是重新校准和导出 NVFP4 的源权重，仅用于量化，没有纳入 BF16 推理基准。FP8 下载没有必要，已停止并清理 877,785,088 字节的未完成权重文件，没有使用 FP8 权重做推理。

源模型固定为腾讯 `tencent/Hy-MT2-1.8B` revision `9a341cd1b679d3efd23b46e847b01745a71ed792`，`model.safetensors` SHA-256 为 `29e9117a44c79f81857613601968ff482d8a23c2d6736a1710bba9e5ca4762e5`。抽查 embedding、第一层 Q 和末层 FFN down 三个张量与原 GGUF 的 BF16 源值一致；这不是完整模型逐字节等价证明。

量化用 llm-compressor 0.13.0 的 NVFP4，256 条 OPUS-100 校准文本，包含项目实际提示和参考译文，最长 4096；与 256 条留出评测文本分离。主模型保留共享输入 embedding / lm_head 为 BF16，线性主干为 4 位，safetensors 文件约 1.27 GiB（现有专用 GGUF 约 0.94 GiB）。另测解除权重共享、将 lm_head 也量化的版本；输入 embedding 仍为 BF16。量化方案应以实际模块精度描述，不能声称每个张量都是 4 位。

## 原生 Windows 兼容处理

vLLM 0.29 默认将此架构映射到 Transformers 模型路径，本机该路径不能正常完成所需的 CUDA Graph 捕获。仓库插件显式注册 vLLM 自带的原生 Hunyuan 实现，并适配其过时的 `AutoWeightsLoader(skip_prefixes=...)` 调用；只在 `HYMT_VLLM_NATIVE=1` 时启用，并限制到经过测试的 0.29.0 版本。

FlashInfer 的采样 JIT 需要 `ninja`、MSVC 编译器和正确的 `LIB` 环境。只安装 Python wheel 不够；`scripts/vllm_windows_env.ps1` 加载 C++ Build Tools 环境。首次编译与首次采样延迟单独记录，不能隐去。

本轮使用一个 API 前端进程。检查发现这版社区包的多 API 前端入口会无条件使用 `socket.SO_REUSEPORT`，而本机原生 Windows Python 没有该常量，因此没有把多前端进程列为可用优化。HTTP 结果包含标准库线程客户端、传输、SSE 处理及服务端开销，不是单独测量 GPU 容量；未锁定 GPU 频率，桌面占用也是波动来源。

## 选择与复现

常驻推理的速度优先候选是 **NVFP4 + 原生 Hunyuan 注册插件 + CUTLASS + 默认 CUDA Graph/编译设置 + 2048 预填充预算**。保持 `stream_interval=1`，不额外启用本轮无收益的融合或输出层量化。若更重视这组留出集的质量表现，改用 Marlin W4A16：权重仍是 4 位，HTTP 吞吐仅略低，质量分数更接近现有版本。

这些收益支持继续评估常驻后端迁移；频繁冷启动的短批任务需要把初始化时间计入决策。本轮没有改动桌面端和运行包的默认后端。测试结束后已恢复原来端口 18080、32 并发、2500 上下文、2048 batch、256 ubatch、F16 KV 的 llama.cpp 服务。

以下在项目根目录的 PowerShell 中执行。需要本机 CPython 3.12、`uv`、Visual Studio C++ Build Tools 与 CUDA Toolkit；两个独立环境避免量化与推理依赖冲突。

```powershell
uv venv .local/vllm-win --python 3.12
uv pip install --python .local/vllm-win/Scripts/python.exe -r requirements-vllm-windows.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv pip install --python .local/vllm-win/Scripts/python.exe -e scripts/vllm_native_plugin
uv venv .local/quant-win --python 3.12
uv pip install --python .local/quant-win/Scripts/python.exe -r requirements-vllm-quant.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

校准文件每行包含 `id`、`text`、`target_lang`、`reference`；评测与校准不得交叠。下载源目录同时保存 `download-provenance.json`（来源仓库与固定 revision）。本次本机数据位于 `results/opus-v2/`，不随精简仓库发布，数据集哈希保存在公开结果中。可在安装了 `pyarrow` 的环境中运行 `scripts/prepare_opus_calibration.py --output-dir results/opus-v2` 生成固定分集；不要覆盖已有实验数据。

```powershell
.local/quant-win/Scripts/python.exe scripts/quantize_vllm_nvfp4.py --source models/Hy-MT2-1.8B --calibration results/opus-v2/calibration.jsonl --evaluation results/opus-v2/evaluation.jsonl --output models/Hy-MT2-1.8B-NVFP4-vllm
. scripts/vllm_windows_env.ps1
.local/vllm-win/Scripts/python.exe scripts/serve_vllm.py --parallel 32 --kv-cache-dtype auto
```

另开终端执行同一 HTTP 客户端；把并发改为 256 时也要重启对应 256 配置的服务。每次使用新的输出目录。测试前停止其他 GPU 推理服务。

```powershell
.venv/Scripts/python.exe scripts/benchmark_backend.py --backend vllm --url http://127.0.0.1:18083 --concurrency 32 --stream --output results/my-vllm-p32
```

复现核心主对照，需要先按 [编译说明](BUILD.md) 准备项目源码、`build/production` 导入库与 `bin` 推理 DLL。以下为独立计时插桩，不修改生产源码或 DLL；构建和结果目录必须使用新名字。先停止其他 GPU 推理服务，矩阵脚本不会代为停止/恢复服务。

```powershell
. scripts/vllm_windows_env.ps1
.venv/Scripts/python.exe scripts/build_core_benchmark.py --output .local/my-core-build
if ($LASTEXITCODE -ne 0) { throw 'Core benchmark build failed' }
.venv/Scripts/python.exe scripts/benchmark_core_matrix.py --native .local/my-core-build/hy-batch-core.exe --output results/my-core-matrix --trials 3 --repeats 3
```

计时补丁为 `patches/hy-batch-core-benchmark.patch`，只应用到独立源码副本。输出包含逐轮 JSON、提示 token ID、译文及 GPU 遥测。原生主指标读 `core_wall_s` / `core_completion_tokens_per_second`；其继承的旧 `wall_s` 说明不适用于新核心字段。vLLM 使用 `--pretokenized` 下的 `wall_s`。预处理时间单独记录，但原生包含输入文件解析、vLLM 只含提示构造/分词，两者的预处理计时不用于速度比较。

项目管理的原服务可用 `powershell -NoProfile -File scripts/stop-server.ps1 -Label server` 停止；停止前保存 `results/server.config.json`，测试后按保存参数重新启动。本轮恢复命令如下，仅对应本轮测试前的配置：

```powershell
.venv/Scripts/python.exe scripts/serve.py --parallel 32 --context 2500 --batch 2048 --ubatch 256 --port 18080 --label server --background
```

单独复测 vLLM 核心批量可导出相同配置：

```powershell
.venv/Scripts/python.exe scripts/serve_vllm.py --parallel 256 --kv-cache-dtype auto --write-config results/my-vllm-p256.json
. scripts/vllm_windows_env.ps1
.local/vllm-win/Scripts/python.exe scripts/benchmark_vllm_offline.py --config results/my-vllm-p256.json --pretokenized --output results/my-vllm-batch-p256
```

`--kernel marlin` 切换 W4A16，`--fused` 打开额外融合，`--batch-tokens` 调整调度 token 预算。复现本页原 BF16 KV 表格时必须保留 `--kv-cache-dtype auto`；新启动入口默认采用 INT8 KV，Marlin 加 INT8 KV 不属于本页已测质量组合。这些开关的收益以结果表为准。生成预算由客户端控制，不应为了提高分数缩短译文。

本页 BF16 KV 对照预算为 32 并发 3 GiB、256 并发 6 GiB，足以处理本次短文本集；6 GiB 不保证 256 条都占满 1024 上下文时仍无抢占。新 INT8 默认按相同 token 容量降低预算。长文本场景需提高 `--kv-gib` 或降低并发后复测。显存比较使用设备峰值减启动前基线的估计，包含初始化、工作区与 CUDA Graph；它不是精确的进程独占显存。
