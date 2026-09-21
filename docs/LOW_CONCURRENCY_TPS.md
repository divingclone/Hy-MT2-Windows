# 32 并发 CUDA Graph 优化

0.2.4 的工作区、跑分和便携包使用同一正式运行库：vLLM 0.29.0+cu132、定制 PyTorch 2.11.0+cu130。CUDA DLL 的 SHA-256 为 `53ce23444795d7118d99f9231567e7f264ff4cac4bb714584454f84cd1bf3f02`，发布校验核对二进制与跑分身份一致。

## 改动

服务最大活动序列为 32 时，vLLM 默认 CUDA Graph 覆盖 64 token。新请求的预填充与其他请求的解码混合后可能超出该范围，回退到未捕获执行。本次共享配置改为 `min(batch_tokens, max(256, min(2 * parallel, 512)))`：32 并发覆盖 256，128 并发仍为 256，256 并发仍为 512。

模型精度、INT8 KV、采样、活动序列数及调度预算不变。收益依赖混合请求负载，不代表纯解码内核加速；原本服务已配置 128 并发时没有额外覆盖收益。

## 性能

RTX 5090 / Ryzen 7 9800X3D / Windows，固定 512 条中英请求，服务和 aiohttp 客户端均为 32 并发，非流式 API。上下文与调度预算均为 2048 token，KV 池 49,152 token（1584 MiB）；温度 0.7、top-p 0.6、top-k 20、重复惩罚 1.05、种子 `42 + index`，最大生成 512。

新旧各三个独立服务进程，逐组反转顺序；每进程先跑一轮完整负载，再跑三轮热测。先取进程内中位数，再取三个进程中位数。计时含完整 HTTP 响应和 JSON 解析，排除初始化及写盘；无显存轮询、GPU profiler 或并行编译。24 轮、12,288 条请求全部成功，无截断。

| 指标 | 原配置 64 | 新配置 256 |
| --- | ---: | ---: |
| 输出 TPS | 4,758.17 | 7,700.33 |
| 进程 TPS 范围 | 4,679.81–4,777.65 | 7,640.08–7,715.92 |
| P95 完整响应时间 | 551.94 ms | 329.98 ms |
| 服务就绪时间中位数 | 30.66 s | 31.63 s |

TPS 增加 **61.8%**，P95 降低 **40.2%**。图捕获增加启动工作；服务启动受缓存和系统波动影响，不能只靠中位数断言成本不变。[性能数据](../benchmarks/graph-coverage-c32.json)保留逐轮记录、捕获日志舍入值和输出哈希。[历史官方对照](API_OPTIMIZATION.md)来自独立实验，不混算加速比。

## 显存

使用 WDDM `GPU Process Memory` 汇总测试服务进程树，排除既有服务和桌面应用。目标间隔 100 ms，实际最大间隔 140 ms；新旧各两个独立进程，顺序 64 / 256 / 256 / 64，每进程冷、热各 512 条请求，4,096 条全部成功。

| 阶段 | 原配置 | 新配置 | 差值 |
| --- | ---: | ---: | ---: |
| 请求阶段峰值（两次中的最大值） | 3,795.94 MiB | 3,899.94 MiB | **+104 MiB（2.74%）** |
| 使用已有编译缓存的全过程峰值 | 3,795.94 MiB | 3,899.94 MiB | +104 MiB |
| 首次重新编译的全过程峰值 | 4,113.94 MiB | 4,113.94 MiB | 0 |
| 共享 GPU 内存峰值 | 86 MiB | 86 MiB | 0 |

首次重编译暂存分配高于后续请求阶段，不能把请求阶段的 +104 MiB 当成所有启动条件的全过程峰值增量。采样可能漏掉更短的瞬时峰值，不是整张卡的峰值保证。逐进程记录和一次未进入请求阶段的端口冲突重试见 [显存数据](../benchmarks/graph-coverage-memory-c32.json)。

## 验证与复现

同一正式运行库下，受控离线调度两配置各跑冷、热两轮，共 2,048 条请求，1,024 组新旧输出逐 token（含 EOS）一致。仅验证进程设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；产品保持默认多进程，不保证在线逐次译文相同，也不将文本一致性当作语义质量评分。

另测 256 并发两轮共 1,024 条请求、INT4 / Marlin 64 条请求，全部成功。84 项 Python、3 项前端、Rust 进程清理和真实 GPU 鉴权、流式、重启与清理检查通过。256 并发属于运行回归，不是新的加速比；RTX 30/40 尚未实卡验证。详见 [验证摘要](../benchmarks/graph-coverage-validation.json)。

复用同版正式便携包的运行库，保持其他 GPU 服务空闲，使用新的输出目录：

```powershell
.\runtime\vllm\python.exe scripts/benchmark_graph_coverage.py --output results/my-graph-comparison --trials 3 --repeats 3
.\runtime\vllm\python.exe scripts/summarize_graph_coverage.py --input results/my-graph-comparison --output results/my-graph-summary.json
.\runtime\vllm\python.exe scripts/benchmark_graph_memory.py --matrix results/my-graph-comparison/matrix.json --output results/my-graph-memory
```

详细记录保留在忽略的 `results/`，公开摘要不含用户路径、PID、提示或译文。更新到 0.2.4 后重启服务生效。
