# 历史同步客户端：官方 llama.cpp 与 vLLM 非流式 API

本报告保留 2026-09-20 的同步 urllib 客户端测量：未修改的官方 llama.cpp Windows CUDA 发布包、腾讯官方 Q4_K_M，比较 Windows 原生 vLLM 部署的 NVFP4 + INT8 KV。通过 `POST /v1/chat/completions`、`stream: false` 获取完整译文，计入 HTTP 和服务端处理。客户端随后已优化，README 现行主对照见 [API 优化与异步客户端复测](API_OPTIMIZATION.md)。历史核心引擎与 SSE 结果不用于计算本表收益。

## 测量结果

| 并发 | 方案 | 输出 token/s | 进程中位数范围 | 请求/s | 512 请求总耗时 | P95 完整响应 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 官方 Q4_K_M / F16 KV | 1,247.29 | 1,168.15–1,257.97 | 19.04 | 26.886 秒 | 2.120 秒 |
| 32 | vLLM NVFP4 / INT8 KV | 3,701.35 | 3,224.66–3,779.05 | 57.56 | 8.895 秒 | 0.785 秒 |
| 256 | 官方 Q4_K_M / F16 KV | 1,327.33 | 1,304.29–1,352.66 | 20.28 | 25.244 秒 | 16.283 秒 |
| 256 | vLLM NVFP4 / INT8 KV | 4,212.79 | 4,122.64–4,231.76 | 65.59 | 7.806 秒 | 6.573 秒 |

32 / 256 并发输出吞吐分别为官方方案的 **2.97 / 3.17 倍**，请求吞吐为 **3.02 / 3.23 倍**；512 条任务总耗时缩短 **66.9% / 69.1%**，P95 完整响应时间缩短 **63.0% / 59.6%**。范围是三个独立进程热测中位数的最小值到最大值，不是置信区间；较慢的复测同样纳入聚合。

有效矩阵共 48 轮、24,576 条请求，全部成功正常结束，无空译文、错误或截断，客户端活跃任务峰值均达到指定并发。另有一次客户端错误导致的一轮无效测试，记录及处理见下文；含该轮总共尝试 25,088 条请求、1 条失败。逐轮指标、输出哈希及失败记录见 [公开 JSON](../benchmarks/official-nonstream-api.json)。

启动与第一轮负载单列如下，均为三个进程的中位数：

| 并发 | 方案 | 服务就绪时间 | 第一轮 512 请求总耗时 |
| --- | --- | ---: | ---: |
| 32 | 官方 Q4_K_M / F16 KV | 1.609 秒 | 27.442 秒 |
| 32 | vLLM NVFP4 / INT8 KV | 35.687 秒 | 9.993 秒 |
| 256 | 官方 Q4_K_M / F16 KV | 1.594 秒 | 25.634 秒 |
| 256 | vLLM NVFP4 / INT8 KV | 40.157 秒 | 7.617 秒 |

vLLM 启动更久；本次第一个 32 并发进程的服务就绪时间达 92.891 秒，首轮负载耗时 18.320 秒。后续进程复用磁盘编译缓存，不能把中位数视为全新安装的首次启动保证。热测收益适用于服务常驻场景，不包含频繁启动/退出的成本。

## 官方基线的身份

- llama.cpp 官方稳定版 [v0.4.1](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.1) 的 `nightly-tag.txt` 指向 [b10964](https://github.com/ggml-org/llama.cpp/releases/tag/b10964)。使用该标签的 `llama-b10964-bin-win-cuda-13.3-x64.zip` 与 `cudart-llama-bin-win-cuda-13.3-x64.zip`；二进制自报 `0.4.1-dev (build 10964, commit b29c606e2)`。这是稳定发布指定的官方构建，没有自行编译或打补丁。
- 模型是 [腾讯官方 Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/tree/a0c709d9fac510f2c807aa3af52872340dc37a4a) 的 `Hy-MT2-1.8B-Q4_K_M.gguf`。不重新量化、不转换布局、不替换模型张量。
- 下载时校验官方资产 SHA-256，保存 ZIP 解压文件逐项哈希；运行矩阵前再次验证二进制及 GGUF。完整 URL、revision、大小、哈希和文件清单收录在公开 JSON 的 `official_baseline`。
- GGUF SHA-256：`dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699`。

开启全 GPU offload、Flash Attention、并发槽位和共享 KV 等均使用原版公开命令行参数。这里“未修改”指模型和程序文件未改动，不代表不设置并发等运行参数；旧项目的 llama.cpp NVFP4/Q8 定制内核不参与本次对照。

## 环境和配置

2026-09-20，Windows x64 原生，RTX 5090 32 GiB、驱动 596.36、AMD Ryzen 7 9800X3D（8 核 / 16 线程）。同一张 GPU 串行运行两种服务，测试期间停止原有翻译服务；每组结束后销毁服务进程树。桌面系统仍在运行，结果只代表这台机器和该工作负载。

| 条件 | 官方 llama.cpp | 本项目 vLLM |
| --- | --- | --- |
| 程序 | 上述官方 CUDA 13.3 Windows 构建 | 社区 Windows wheel 0.29.0+cu132、PyTorch 2.11.0+cu130、本项目兼容插件 |
| 权重 | 腾讯 Q4_K_M | 本项目 NVFP4，CUTLASS W4A4；embedding/head 保留 BF16 |
| KV | 官方默认 F16 | 项目默认动态 `int8_per_token_head` |
| 单请求上下文 | 2048 token | 2048 token |
| 调度 token 上限 | batch / microbatch 均 2048 | max_num_batched_tokens = 2048 |
| 最大活跃请求 | 32 / 256 | 32 / 256 |
| 共享 KV token 容量 | 49,152 / 98,304 | 49,152 / 98,304 |
| 提示缓存 | `cache_prompt=false`，RAM cache = 0 | prefix caching 关闭 |
| 接口 | 本机 HTTP，非流式完整 JSON | 本机 HTTP，非流式完整 JSON |

两端 KV **token 容量相同，字节占用不同**：官方 F16 约 3 / 6 GiB，项目 INT8 含 scale 约 1.547 / 3.094 GiB。共享池不为每个活跃请求独占预留完整 2048 token。vLLM 直接使用发布程序的环境及服务启动函数，运行时来自瘦身后的打包暂存目录，使用单个 API 前端；不以离线批处理替代 API。

本项目 NVFP4 模型来自 [独立原始文件仓库](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-vLLM/tree/2fa2734cffce6dbea70e442095ba1c0f539d23e5)，固定 revision 和各文件哈希保存在公开 JSON 中。

## 请求和计时口径

- 输入：[固定中英双向案例](../scripts/benchmark_cases.json)，64 条源文各重复 8 次，每轮共 512 请求，所有请求具有独立 ID。它是可复现吞吐工作负载，不代表生产流量的文本长度或到达时间分布，也不用于翻译质量评分。
- 两端使用项目 `translation_prompt` 生成的相同 user 消息，由各自模型 tokenizer/chat template 处理。温度 0.7、top_p 0.6、top_k 20、min_p 0、重复惩罚 1.05、seed = 42 + 请求序号，最多生成 512 token。不同实现的采样结果不要求逐字相同。
- 客户端为 Python 标准库 HTTP 请求，匹配项目翻译客户端的方式；每请求单独连接、不使用连接池，关闭系统代理。32 / 256 个工作线程提前创建，在统一起点释放请求；每个线程收到完整响应后才处理下一条。`observed_peak_in_flight` 核验的是客户端活跃任务峰值，包含请求构造和响应解析，不能证明同样数量的 HTTP 请求已送达服务端，更不能证明 GPU 同时解码了 32 / 256 条。
- **总墙钟**：从统一释放到最后一个完整 JSON 解析完成，包含客户端请求构造、HTTP、服务端排队、分词、prefill、decode、采样及响应序列化。排除客户端线程创建/退出、模型加载/JIT 和结果文件写盘。512 条任务的客户端等待时间计入总墙钟。
- **输出 token/s** = 成功完整响应的 `usage.completion_tokens` 总和 / 总墙钟；不估算 token，也不加输入 token。
- **请求/s** = 512 / 总墙钟。两个模型的输出长度可能不同，因此同时展示请求吞吐，避免只靠 token/s 判断实际翻译速度。
- **P95 完整响应时间**：从某请求开始处理到完整 JSON 解析结束，包含服务端排队，排除该请求尚未获得客户端工作线程的等待。包含客户端等待的 `end_to_end_s` 另存逐轮 JSON。非流式客户端无法观察首 token 到达，因此 TTFT 留空，不能用该 P95 代替 TTFT。

每种“后端 × 并发”启动 **3 个独立服务进程**，交错后端及并发顺序；每个进程先运行一整轮 512 请求作为首次负载，再测 3 轮。表格对每个进程的 3 个热测值取中位数，再对 3 个进程中位数取中位数。P95 也按各轮 P95 用相同规则聚合，不是将全部请求合并计算一个 P95。

“预热”在这里指同一服务进程的第一轮完整负载，使实际请求用到的执行路径和缓存完成首次准备；它不是提前缓存译文。首次负载及服务就绪时间单独保留，均不混入热测表。独立进程会复用磁盘编译缓存，因此服务就绪时间也不代表全新安装首次启动。

## 失败记录

第二组官方 256 并发首次负载中，客户端有 1 条请求返回 Windows `WinError 10055`（套接字缓冲区不足或队列已满），其余 511 条完成。该轮整体判为无效，原始记录保留；未从该轮仅抽取成功请求计算主表。续测在配置之间加入 120 秒连接回收间隔，间隔不计入任何性能指标，并重新启动该配置完成完整复测。详见公开 JSON 的 `failed_attempts`、`cooldown_history`。

## 复现

需已有项目原生 vLLM 运行时和 NVFP4 模型，停止其他 GPU 推理服务，确保测试端口空闲。在项目根目录执行：

```powershell
.\runtime\vllm\python.exe scripts/prepare_official_baseline.py
.\runtime\vllm\python.exe scripts/benchmark_official_api.py --client urllib --output results/my-official-api
.\runtime\vllm\python.exe scripts/summarize_official_api.py --input results/my-official-api --output results/my-official-api-summary.json
```

默认运行时根目录是发布暂存目录 `desktop/src-tauri/resources/payload`；若使用源码根目录的完整运行时，可指定 `--runtime-root .`，但应记录其与本报告瘦身运行时的差异。输出目录必须全新；中断后 `--resume` 校验原配置并继续，保留已完成组和失败记录。测试脚本清理自己启动的服务，不会自动管理用户原先运行的服务。

单独测已有服务时：

```powershell
.\runtime\vllm\python.exe scripts/benchmark_backend.py --backend vllm --client urllib --url http://127.0.0.1:18080 --concurrency 32 --repeats 3 --output results/my-api-32
```

默认 `stream: false`。只有显式增加 `--stream` 才会使用 SSE，该结果不属于本报告主表。官方完整启动参数及 vLLM 参数保存在本机 `results/official-nonstream-api/matrix.json`；公开摘要保留其配置意义及官方文件身份。复测结果会受硬件、驱动、系统负载和请求长度影响。

本报告比较的是两套实际部署方案：权重格式、激活精度、KV 精度、调度和 HTTP 实现都不同。收益不能单独归因于 vLLM 内核，也不证明两种量化质量相同。以未量化 BF16 输出为参照的质量分析见 [保真度评测](TEACHER_FIDELITY.md)；旧核心吞吐和 5426 token/s 等历史数字见 [KV 报告](VLLM_KV.md)，计时边界不同。

## 核心 TPS 与 API TPS 差额的后续核查

复核本次 vLLM 的九轮热测原始时间戳发现：256 并发下，按输入顺序前 256 条任务的客户端开始时间 P95 为释放后 **2.359–3.056 秒**（九轮中位数 **2.739 秒**），这一组最后一个任务开始时间为 **2.947–3.242 秒**。这里的“开始”在客户端函数进入时记录，尚未构造、发送 HTTP 请求；并不是服务端收到请求的时间。即使工作线程提前创建，共同释放也没有让请求同时进入引擎。

此前 256 并发核心测试把全部 512 条预分词请求交给 `LLM.generate()`，热测总耗时约 1.381 秒。当前客户端仅释放前一组任务就耗时更久，说明压测端的调度/请求处理已经明显改变了供给节奏。单次客户端任务占用工作线程还包括构造请求、建连、等待和 JSON 解析；客户端活跃峰值不能用作服务端批大小的证据。

因此本表衡量的是**当前同步线程客户端到完整译文的端到端表现**，不是优化客户端之后的 HTTP 服务吞吐上限。不能把 23,795 → 4,213 token/s 的全部差额归为 HTTP 协议、Python 服务端或网络传输的成本。历史核心测试还使用 1024 上下文，本次为 2048，且使用不同时间的开发/打包运行环境，尚未做同环境、同供给节奏的隔离消融。

已核对本机实现：API 路径使用 vLLM 自带的 Python FastAPI/Uvicorn，Windows 入口使用 Winloop，通过 ZeroMQ 与 EngineCore 通信；桌面 UI 不在请求转发路径中。非流式请求的采样参数设为 `FINAL_ONLY`，没有对客户端逐 token 发送 SSE。客户端线程调度、重复建连，以及服务端处理、IPC、引擎实际批大小各占多少耗时，仍需分层测量；当前证据不能给出其独立百分比。下一步应先比较异步连接池客户端，再以相同请求供给方式比较直接 AsyncLLM 与 HTTP，并记录服务端等待/运行请求数。
