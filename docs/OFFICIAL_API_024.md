# 0.2.4 与官方 llama.cpp 的 API 对比

2026-09-21，RTX 5090 / Ryzen 7 9800X3D / Windows 原生。两端通过同一 aiohttp 连接池调用 `POST /v1/chat/completions`，`stream: false`；计时包含完整请求和 JSON 响应解析，排除服务启动及输出写盘。

| 并发 | 部署方案 | TPS（输出 token/s） | 显存峰值增量（GiB） |
| --- | --- | ---: | ---: |
| 32 | 官方 llama.cpp + Q4_K_M / F16 KV | 1,351.39 | 5.01 |
| 32 | 本项目 vLLM + NVFP4 / INT8 KV | 7,391.52 | 3.82 |
| 256 | 官方 llama.cpp + Q4_K_M / F16 KV | 1,500.00 | 8.16 |
| 256 | 本项目 vLLM + NVFP4 / INT8 KV | 20,462.47 | 5.89 |

32 / 256 并发的输出 TPS 分别为官方基线的 **5.47 / 13.64 倍**。这是完整部署方案比较，包含权重量化、激活、KV 精度、调度与接口差异，不代表纯内核收益或质量等效。

## 条件

- 官方程序为固定版本 b10964（稳定版 v0.4.1 对应构建），腾讯官方 Q4_K_M。二进制与模型逐文件 SHA-256 校验，未经修改；来源与哈希保留在 [机器可读结果](../benchmarks/official-api-release024.json)。
- 本项目使用发布包的正式 vLLM 0.29.0+cu132、定制 PyTorch 2.11.0+cu130，NVFP4 / CUTLASS、INT8 KV。CUDA 后端身份与 [优化验证](../benchmarks/graph-coverage-validation.json) 相同。
- 固定 64 条中英双向源文本重复八次，共 512 请求；上下文 2048，调度预算 2048。32 / 256 并发的共享 KV 容量分别为 49,152 / 98,304 token，两端相同；不为每条请求独占分配完整上下文。
- 温度 0.7、top-p 0.6、top-k 20、重复惩罚 1.05、逐请求种子 `42 + index`、最大生成 512。关闭 prompt/prefix 缓存，模型各使用自己的 tokenizer 与模板。
- 每配置三个独立服务进程，各先跑完整预热，再测三轮。逐组反转后端和并发顺序；12 个服务进程、48 轮、24,576 条请求全部成功，无截断。先取进程内三轮中位数，再取三个进程中位数；原始首轮和较慢复测均保留。

## 显存口径

使用 nvidia-smi 每 100 ms 采样整张卡的已用显存，以服务启动至停止的最大值减去启动前基线，再取三个独立进程的中位数。包含初始化、CUDA Graph、KV 与工作区；存在桌面噪声和采样间隙，不能视为精确的进程独占峰值。

这是 README 中新旧方案统一使用的口径；[图覆盖优化报告](LOW_CONCURRENCY_TPS.md)另用 WDDM 进程树计数器区分启动和请求阶段，两种显存口径不可混算。本矩阵计时含遥测，独立图覆盖消融不含遥测，因此不混合它们的 TPS 计算加速比。

## 复现

准备 [官方基线](../scripts/prepare_official_baseline.py) 和同版正式运行库后，停止其他 GPU 推理负载，运行：

```powershell
.\runtime\vllm\python.exe scripts/benchmark_official_api.py --output results/my-official-comparison --runtime-root desktop/src-tauri/resources/payload --parallels 32 256 --trials 3 --repeats 3 --cooldown 2
.\runtime\vllm\python.exe scripts/summarize_official_api.py --input results/my-official-comparison --output results/my-official-summary.json
```

原始记录保留于 `results/official-api-release024/`，发布的精简摘要不含本机路径和逐条译文。仅 RTX 5090 实测；小输入集不能覆盖所有生产负载。启动时间、逐轮分位数、进程波动和输出哈希见机器可读结果；语义质量另见 [BF16 参考保真度](TEACHER_FIDELITY.md)。
