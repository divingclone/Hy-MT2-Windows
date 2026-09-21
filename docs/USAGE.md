# vLLM 使用说明

运行前准备原生 Windows vLLM 完整运行时和对应显卡的 4 位模型，具体要求见 [README](../README.md)。`setup-model.cmd` 首次从 Hugging Face 下载对应显卡的一个模型，逐文件校验后原子发布目录；已有 checkpoint 自动复用。旧 GGUF 不兼容。

## API

`start-server.cmd -Background` 等待服务就绪；`stop-server.cmd` 停止服务和所有推理子进程。Base URL 为 `http://127.0.0.1:18080/v1`，模型名 `hy-mt2`。支持 `/v1/chat/completions` 普通与流式响应；`/health` 以 HTTP 200 表示健康，响应体可为空。

命令行可通过环境变量 `VLLM_API_KEY` 配置密钥。桌面端默认开启密钥，保存到应用数据目录，日志会脱敏；不要将密钥提交到 Git。

```powershell
.\start-server.cmd -Background -Parallel 256 -ContextPerSlot 4096 -KVGib 4.125 -MemoryPercent 60
.\stop-server.cmd
```

参数：`-Profile auto|fast|quality|compat`，`-KVCacheDtype int8_per_token_head|bfloat16|fp8_per_token_head`，`-BatchTokens 2048`，`-Gpu`（索引或 UUID），`-Port`，`-Label`。auto 为 RTX 50 选择 fast，为 RTX 30/40 选择 compat。fast 使用 NVFP4 CUTLASS，quality 使用 NVFP4 Marlin，compat 使用普通 GPTQ INT4 Marlin；所有部署模式均为 4 位权重。默认 INT8；Marlin+INT8 不是已完成保真度矩阵中的组合，业务评测后采用。

显存预算包含模型、工作区估计、安全余量和显式 KV 池，不能保证所有 CUDA Graph/长输入场景都不 OOM。32 并发默认池可存 49,152 token，256 并发默认 98,304 token；并发不是满上下文槽位预留。INT8 包含 FP32 动态尺度，相同 token 容量下比 BF16 KV 节约 48.44%。

当前源码扩大了小并发的 CUDA Graph 覆盖：默认 32 并发捕获到 256 token，减少混合预填充时的执行开销；256 并发仍捕获到 512 token。不会提高活动请求上限，较小的 `-BatchTokens` 仍会限制捕获范围。需要重启服务生效，详见 [32 并发优化](LOW_CONCURRENCY_TPS.md)。

### 通过 API 批量翻译

```powershell
.\runtime\vllm\python.exe scripts/translate.py --input-jsonl examples/input.jsonl --concurrency 256 --output results/translations.jsonl
```

服务端同时使用 `-Parallel 256` 才能提供对应的引擎并发上限。此客户端使用 aiohttp 异步请求及连接池，每条请求仍为标准 `stream: false`，完整收到译文后才补充下一条；不是将多条文本合并成一次推理请求。保留输入顺序、ID 和每条种子，错误、截断、超时分别记录，失败不自动重试。依赖已包含在完整运行时中。异步 Python 程序可直接调用 `translate_many_async`，同步程序可调用 `translate_many`。服务启用鉴权时，在客户端进程的 `VLLM_API_KEY` 环境变量中配置对应密钥；仅放入请求头，不写入结果。

第三方客户端也应使用共享连接池，池容量至少匹配目标并发，并关闭本机请求的系统代理。大量同步线程和每条请求重复创建客户端可能成为瓶颈；客户端活跃任务数不等于服务端收到的请求数或 GPU 实际批大小。重复惩罚使用 vLLM 的 `repetition_penalty` 字段。前缀缓存由服务端配置，旧 `--cache-prompt` 参数仅保留命令兼容。

## 离线批量

```powershell
.\translate-batch.cmd examples/input.jsonl output.jsonl --parallel 32 --context 2048 --max-tokens 512
```

使用直接 vLLM 引擎，无 HTTP。模型初始化单列，输入先分词；输入提示加生成预算超过上下文时明确失败，不静默截断。保留顺序/ID，输出 `ok`、`truncated` 和真实 completion token 数；失败或截断返回非零退出码。已存在输出及旁路文件不会覆盖。

## 排查

- 首次启动可能需要数分钟编译；后台启动最多等待 600 秒。
- 编译失败：检查运行包是否完整、路径是否可写；Triton 使用随包编译组件，FlashInfer 使用预编译内核，无需安装 MSVC/CUDA Toolkit。
- OOM：降低并发、调度 token 预算或 KV 预算，关闭其他 GPU 服务；必要时增加总预算比例。
- 模型损坏：使用模型页或 `setup-model.cmd --source 模型包.zip` 导入清单对应的完整 ZIP。也可通过模型页重新下载，按固定提交和 SHA-256 校验。
- 旧版桌面 Q8/F16 设置自动映射到 INT8/BF16，旧 Q4 设置回到 INT8；当前不提供 GGUF 回退后端。
