# 非流式翻译 API：客户端瓶颈定位与优化

本轮保留 Windows 原生 vLLM、4 位 NVFP4 权重、INT8 KV 和标准 OpenAI 非流式接口。主要改动是将随包翻译客户端及默认压测客户端，从同步线程 + 每请求新建 urllib opener，改为 **aiohttp + asyncio、受限异步并发和共享连接池**。服务端仍为 vLLM 自带的 FastAPI/Uvicorn/Winloop，未重写 Web 框架。

## 定位结论

历史 256 并发压测中，前 256 条客户端任务的开始时间 P95 为 2.36–3.06 秒；这时任务尚未构造、发送请求。实际 GPU 批大小不能由客户端活跃任务峰值推出。之前的 4,213 token/s 反映了同步客户端到完整译文的组合表现，不是 vLLM HTTP 服务的上限，详见 [历史同步客户端报告](OFFICIAL_API_BENCHMARK.md)。

在同一服务进程、同一模型及采样参数下，仅替换客户端即可将 256 并发提高到约 2 万 token/s。两种客户端均为 Python，因此证据支持优化请求处理方式，不能把差额归为 Python 语言本身或 HTTP 协议固有成本。aiohttp 同时改变了调度、HTTP 实现和连接管理，本轮没有进一步分离 GIL、线程切换、stdlib 请求构造等因素的独立占比。

接口路径为：客户端 → Python FastAPI/Uvicorn → AsyncLLM → ZeroMQ → EngineCore → GPU 内核。官方说明见 [vLLM 架构](https://docs.vllm.ai/en/latest/design/arch_overview/)；连接池的生命周期与限制见 [aiohttp 客户端文档](https://docs.aiohttp.org/en/stable/client_advanced.html)。非流式请求使用 `FINAL_ONLY` 输出模式，并不逐 token 向客户端发送 SSE。

## 探索方法与选择

固定 512 条中英双向翻译、32 / 256 最大客户端并发，上下文 2048，最大生成 512，采样参数及逐条种子保持一致，关闭前缀缓存。每个候选先测一次完整负载，再测三轮；同进程比较客户端，第二轮反转客户端顺序。探索结果是两轮热测中位数的中位数，不是跨独立进程的置信区间。

最终选择 aiohttp 的标准 asyncio 循环、单 API 前端、2048 调度 token。各候选完整结果、较慢轮次、输出哈希及失败记录见 [优化证据](../benchmarks/api-optimization.json)。

| 客户端 | 32 并发 TPS | 256 并发 TPS |
| --- | ---: | ---: |
| 同步 urllib 线程 | 3,726.55 | 4,584.29 |
| aiohttp + asyncio，连接池 | 4,553.23 | 19,893.18 |
| aiohttp + Winloop，连接池 | 4,417.05 | 18,463.91 |
| aiohttp + Winloop，每次关闭连接 | 未测 | 19,520.40 |

同进程交错筛选中，选定客户端相对旧客户端约为 1.22 / 4.34 倍。该表用于定位客户端差异；下面独立进程的正式对照与它不是同一组统计样本，保留实际波动，不挑选更高的一组作为主结果。

- **asyncio 与 Winloop 客户端**：两者均大幅改善 256 并发；额外换用 Winloop 没有稳定优势。服务端自身的 Winloop 保持原样。
- **连接复用消融**：异步客户端即使关闭 keep-alive 也达到接近吞吐，因此不能将全部增益归给连接复用。保留连接池可以减少建连和端口消耗；不做请求自动重试来掩盖失败。
- **调度预算 1024 / 2048 / 4096**：4096 未显示稳定改善；1024 在隔离复查中仍较慢，保留 2048。最初 1024 探索与 CPU 单元测试有时间重叠，明确排除出参数选择依据，并保留记录，以无并行单元测试的 `api-opt-batch1024-recheck` 代替。
- **双 API 前端**：这版原生 Windows wheel 的多前端启动路径调用缺失的 `socket.SO_REUSEPORT`，启动失败，未产生有效吞吐结果。单前端已接近本轮相同配置的核心实测，不引入额外代理或维护一套新服务框架。
- **同配置核心复查**：使用当前运行时、2048 上下文、相同 KV 容量和 kernel 参数，离线预分词核心的三轮热测中位数约 19,383 token/s。该项只有一个进程，且与 HTTP 依次运行，不能据小幅差异宣称 HTTP 超过核心。历史 1024 上下文的 23,795 token/s 不作为本轮 HTTP 效率百分比的分母。

## 最终官方对照

采用选定客户端后，对未修改的官方 llama.cpp + 腾讯 Q4_K_M 和本项目 vLLM **双方重新测量**，不拿旧同步客户端的官方结果去比较新异步客户端的 vLLM。

每种后端 × 并发采用三个独立进程、每进程首次负载 + 三轮热测，交错顺序，逐级取中位数。两边固定 2048 上下文、2048 调度 token、相同 KV token 容量；官方保留 F16 KV，项目保留 INT8 KV。原始结果的 ID、完成状态、真实输出 token、文件哈希逐轮核验。

2026-09-20–21，RTX 5090 / Ryzen 7 9800X3D，原生 Windows、驱动 596.36。正式结果如下：

| 并发 | 方案 | 输出 TPS | 进程 TPS 范围 | 显存峰值增量 | 显存增量范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| 32 | 官方 Q4_K_M / F16 KV | 1,258.52 | 1,235.05–1,276.54 | 5.13 GiB | 5.07–5.40 GiB |
| 32 | vLLM NVFP4 / INT8 KV | 4,061.30 | 3,904.92–4,283.11 | 4.12 GiB | 3.59–4.30 GiB |
| 256 | 官方 Q4_K_M / F16 KV | 1,418.02 | 1,385.01–1,459.39 | 8.21 GiB | 8.15–8.28 GiB |
| 256 | vLLM NVFP4 / INT8 KV | 18,315.16 | 18,144.55–19,316.53 | 5.91 GiB | 5.85–5.91 GiB |

正式矩阵 48 轮、24,576 条请求全部成功，未出现错误、空输出或截断。官方二进制为 v0.4.1 指定的 b10964，腾讯 Q4_K_M 文件与两份官方 CUDA ZIP 均未修改，启动前复核哈希；vLLM 使用瘦身后的发布暂存运行时。两边共享 KV 池分别为 49,152 / 98,304 token，官方 F16 与项目 INT8 的字节容量不同。

256 并发九轮正式热测中，前 256 条客户端任务开始时间 P95 的中位数降至 **70 ms**（62–75 ms）；请求头准备发送回调 P95 的中位数为 **101 ms**（85–106 ms）。此前同步客户端仅任务开始时间的 P95 就达数秒，供给节奏已有明确改善。77 项单元测试通过；使用发布暂存运行时及实际翻译脚本另外验证 512 条请求全部成功、无截断，顺序、ID 和种子一致，该功能验证不计入性能主表。

显存由 nvidia-smi 每 100 ms 采样，取**整个服务生命周期的设备已用显存峰值减启动前第一帧基线**，包括初始化、CUDA Graph、KV、工作区及桌面噪声，不是精确进程独占值。每个进程取一个峰值增量，再取三个进程中位数。32 并发 vLLM 的桌面基线波动较明显，因此保留 3.59–4.30 GiB 范围，不把中位数当成严格显存上限。

P95 完整响应时间（官方 → 项目）：32 并发 2.133 → 0.653 秒，256 并发 14.938 → 1.212 秒；完整请求吞吐分别为 19.24 → 63.17、21.68 → 285.43 请求/秒。P95 按逐轮分位数再逐级取中位数，不是合并全部请求的 P95。服务就绪时间中位数为官方约 1.6 秒、vLLM 32 / 256 并发约 36.2 / 39.2 秒；vLLM 第一个进程为 75.4 秒。常驻吞吐没有包含启动成本。

完整数值和复测范围见 [异步 API 基准](../benchmarks/official-nonstream-api-async.json)。此处的提升包含模型量化、引擎与 API 实现差异，不代表纯内核收益或翻译质量等价。原始文件与公开摘要均保留较慢复测，不以探索中的最高 TPS 替代正式中位数。

## 计时范围

从开始调度这一轮客户端任务，到所有完整 JSON 响应解析结束。包含请求构造、连接建立、HTTP、分词、服务端排队、推理、响应序列化及客户端解析；不包含服务初始化、事件循环/会话建立及关闭、测量后写盘。每轮使用新会话，轮内复用连接，不预先建立 TCP 连接来降低数字。

P95 为客户端任务获得并发名额后的完整响应时间，包含服务端排队，不包含等待客户端并发名额的时间；512 请求总墙钟包含全部等待。`observed_peak_in_flight` 是客户端活跃任务峰值，不是服务端或 GPU 批大小。新增 `first_group_client_start_s` 与 `first_group_headers_ready_s` 用于审计供给节奏，后者来自 aiohttp 发送请求头前的回调，也不是服务端收包时间。

初始化与首次负载单独保留；“热测”指服务执行路径已经历一轮负载，不预先缓存译文。模型、KV 精度、文本、生成上限和温度均未为提速降低。服务仍支持流式调用，主对照只使用非流式翻译。

## 项目落地与复现

`scripts/translate.py` 默认批量路径使用异步连接池，并提供 `translate_many_async` 给已有异步程序。结果保持输入顺序和逐条种子，HTTP 错误、超时、截断不会计作成功。修正了旧客户端发送 llama.cpp `repeat_penalty` 字段的问题，改用 vLLM 的 `repetition_penalty`；默认仍为 1.05。鉴权通过进程环境变量 `VLLM_API_KEY` 放入请求头，不写入译文或日志。

`http_transport.py` 已加入便携包文件清单，aiohttp/Winloop 原本就在 vLLM 运行时中，本轮固定其依赖版本，没有引入另一套大型运行环境。用法见 [API 翻译说明](USAGE.md)。0.2.2 正式便携包重新构建并包含这些客户端改动；发布前的 cuDNN 依赖裁剪回归另见 [运行包精简](RUNTIME_SIZE.md)。

```powershell
# 先停止其他 GPU 推理服务；脚本只管理自己启动的服务。
.\runtime\vllm\python.exe scripts/prepare_official_baseline.py
.\runtime\vllm\python.exe scripts/benchmark_official_api.py --client aiohttp --cooldown 0 --output results/my-async-matrix
.\runtime\vllm\python.exe scripts/summarize_official_api.py --input results/my-async-matrix --output results/my-async-summary.json

# 同进程客户端对照；默认单 API 前端、2048 调度 token。
.\runtime\vllm\python.exe scripts/benchmark_api_optimization.py --parallel 256 --clients urllib aiohttp aiohttp-winloop aiohttp-winloop-close --rounds 2 --core --output results/my-client-screen
```

已有服务可直接运行 `benchmark_backend.py --client aiohttp`，默认非流式；旧同步路径通过 `--client urllib` 保留。显式 `--stream` 默认使用 urllib，SSE 结果不与本页混算。完整矩阵原始记录保留在忽略的 `results/official-nonstream-api-async/`；本机优化筛选位于 `results/api-opt-*`。
