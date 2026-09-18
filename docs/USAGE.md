# 安装与使用

## 安装

1. 下载本仓库 Releases 中的 `HyMT-Windows-NVIDIA-runtime.zip`，解压到可写目录。源码 ZIP 用于开发，不包含可执行程序和运行依赖。
2. 安装适合显卡的 NVIDIA 驱动，版本至少 580.88。运行包自带 Python、CUDA 和 MSVC 运行库，不需要 pip、WSL、Docker 或另外安装 CUDA Toolkit。
3. 在解压目录打开 PowerShell，运行：

```powershell
.\setup-model.cmd
```

程序检测显卡，从 [模型仓库](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF) 下载所选模型到 `models`。首次下载需要网络，下载完成后可以离线使用。下载支持断点续传，并按清单校验文件大小和 SHA-256；已有正确文件会复用。下载脚本的其他选项见 `setup-model.cmd --help`。

| 配置 | 模型选择 |
| --- | --- |
| `auto`，默认 | 支持的 CC 12.0 显卡优先 NVFP4，其他支持架构使用 Q4_K_M |
| `official` | 腾讯 Q4_K_M 量化权重，经本工程无损布局重排 |
| `fast` | 从 BF16 经 OPUS 中英校准与局部 MSE 尺度搜索量化的 NVFP4，用于本包支持的 CC 12.0 显卡 |

切换配置前先下载对应模型：

```powershell
.\setup-model.cmd --profile official
.\translate-batch.cmd examples\input.jsonl output.jsonl --profile official
```

启动器统一使用 `bin` 中的当前程序，并根据 `models/manifest.json` 选择模型；每次启动都会读取完整模型文件，校验大小和 SHA-256，不符时要求重新安装。显式 `--model` 路径保留用户自选；默认配置不会自动选择历史构建或另一种模型。

## 硬件范围

| 显卡架构 / 常见系列 | CUDA 计算能力 | 自动模型 | 验证范围 |
| --- | --- | --- | --- |
| Turing：GTX 16、RTX 20 | 7.5 | Q4_K_M | 编译覆盖 |
| Ampere：8.0 设备、RTX 30 | 8.0 / 8.6 | Q4_K_M | 编译覆盖 |
| Ada：RTX 40 | 8.9 | Q4_K_M | 编译覆盖 |
| Blackwell：RTX 50 | 12.0 | NVFP4 | RTX 5090 实机验证 |

运行包包含 `sm_75/80/86/89/120a` 原生 CUDA 代码，不包含 PTX。没有列出的计算能力不在本包兼容范围内，`120a` 也不是对未来 GPU 的兼容承诺。CPU 为 x64/SSE2 基线，无额外 AVX2 要求。测试系统为 Windows 11；需要 Windows x64。

```powershell
.\gpu-info.cmd
# 多显卡时，下载和运行均可指定同一 GPU
.\setup-model.cmd --gpu 0
.\translate-batch.cmd input.jsonl output.jsonl --gpu 0
.\start-server.cmd -Gpu 0
```

默认选择兼容且可用显存最多的一张卡，不会自动把一个模型拆到多张 GPU。

## 文件批量翻译

```powershell
.\translate-batch.cmd examples\input.jsonl output.jsonl
```

输入为 UTF-8 JSONL，允许 BOM 和空行。每条请求包含非空 `text`、目标语言 `target_lang`，以及可选 `id`；建议显式提供唯一 ID。输出按输入顺序写出，包含译文、token 计数和结束原因。支持中文文件名。

```json
{"id":"en-1","text":"The meeting starts at 9:30.","target_lang":"Chinese"}
{"id":"zh-1","text":"请勿关闭电源。","target_lang":"English"}
{"id":"ja-1","text":"明日の予約を変更できますか。","target_lang":"English"}
```

可使用 `Chinese`、`English`、`Japanese` 等语言名，也支持 `zh`、`en`、`ja` 和常用中文别名。省略输出路径时，默认生成 `输入文件名.translated.jsonl`。

输出旁会生成 `.summary.json`、`.config.json` 和 `.log`。重点查看 `truncated`、`empty_outputs` 及退出码；输出截断或为空时程序返回非零状态。配置和日志可能含本机路径与设备标识，分享前自行脱敏。

默认使用官方采样设置：temperature 0.7、top-p 0.6、top-k 20、repeat penalty 1.05、min-p 0；每条请求使用 `42 + 请求序号` 的独立种子。可用 `--seed` 更改种子，`--greedy` 使用贪心采样。文件模式默认 GPU 批量候选筛选、1 个 CPU 尾部采样线程；`--cpu-sampling` 切回 CPU 采样。

## 显存与长文本

**优先保留自动配置。** 自动并发使用启动时可用显存计算，而不是显卡标称容量；文件模式最高 256，服务最高 128。模型文件大小、所选 KV cache 精度、工作区和安全余量都会计入估计。小于 6 GiB 可用显存时自动采用较小物理批量。

本模型每条 1024 token 上下文的 FP16 KV 约占 **64 MiB**；128 条约 8 GiB，256 条约 16 GiB，另加模型和工作区。4 位模型不等于 4 位 KV cache。

默认 K/V 均为 F16。Q8_0 每条 1024 token 的 KV 约占 **34 MiB**（已包含分块尺度），可降低显存占用，但吞吐和译文会变化。两种接口使用相同选项：

```powershell
.\translate-batch.cmd input.jsonl output.jsonl --cache-type-k q8_0 --cache-type-v q8_0
.\start-server.cmd -CacheTypeK q8_0 -CacheTypeV q8_0
```

建议 K/V 选择相同类型。Q8 写入融合默认启用；改变 attention 累加顺序的实验 subwarp 不在默认运行路径中。具体实测见 [性能报告](PERFORMANCE.md)。

以下是手动尝试时的保守起点，假设 GPU 基本空闲、上下文 1024、最大生成 512。**除 RTX 5090 外未逐卡实测**，不是保证可用或最优的配置；其他程序占用较多时应降低并发。

| 标称显存 | 建议先试并发 | 建议物理批量 `--ubatch` |
| --- | ---: | ---: |
| 4 GB | 4 | 256 |
| 6 GB | 8 | 256 |
| 8 GB | 16 | 512 |
| 12 GB | 32 | 512 |
| 16 GB | 64 | 1024 |
| 24 GB | 128 | 2048 |
| 32 GB | 256（API 上限 128） | 2048 |

```powershell
# 较小显存
.\translate-batch.cmd input.jsonl output.jsonl --parallel 4 --context 1024 --ubatch 256
# 长文本：上下文扩大一倍时，可先将并发减半
.\translate-batch.cmd input.jsonl output.jsonl --parallel 32 --context 2048 --max-tokens 1024
```

上下文包含翻译指令、原文和生成译文，单位为 token，不是字符。程序检查提示和输出预算；超限时应分段，或同时增大 `--context` 与所需的 `--max-tokens`。显式设置超过保守显存预算会报错，不会静默降低用户指定的并发。

当前程序的速度和显存以 [性能报告](PERFORMANCE.md) 的同期实测为准；并发、上下文及 K/V 类型共同决定占用。历史设备观测保留在 [显存证据](../benchmarks/rtx5090-vram.json)，不作为新配置的容量保证。

请求可以按实际业务顺序随机长短混合输入。默认文件批处理按波次执行，会等待该波全部结束；GPU 前缀路径最后不足一波仍保留填充槽位，实际请求很少时程序会降低初始并发。长短相差很大时吞吐可能下降；保留完整原文和所需生成预算，不能使用固定长度跑分决定生产预算。

## 本地网页与 API

```powershell
.\start-server.cmd
.\stop-server.cmd
# 停止已有服务后，使用长文本配置重新启动
.\start-server.cmd -Parallel 32 -ContextPerSlot 2048 -Profile official
```

网页为 <http://127.0.0.1:18080>，健康检查 `/health`；兼容 OpenAI 的接口为 `http://127.0.0.1:18080/v1`，模型名 `hy-mt2`。服务只监听本机，日志在 `results/server.*.log`。

附带客户端会使用该模型的翻译提示与采样参数：

```powershell
.\scripts\run-python.cmd scripts\translate.py "Please keep your ticket." --target Chinese
```

接入其他客户端时，通过 `/v1/chat/completions` 发送一条 user 消息，例如 `Translate the following text into Chinese. Note that you should only output the translated result without any additional explanation:\nPlease keep your ticket.`，不要额外添加通用聊天 system prompt。可按上面的官方采样值配置客户端。

API 用于应用接入；最高吞吐数字来自文件批处理，GPU 批量前缀目前未接入 HTTP 服务。测量文件吞吐前先停止服务及其他 GPU 工作负载。

服务关闭 context shift，避免生成过程中静默丢弃前文。客户端应为完整提示与所需输出保留上下文，并检查 HTTP 错误和 `finish_reason`；超出上下文的输入需要明确处理。

## 常见问题

- **找不到 NVIDIA GPU / 驱动过旧**：先运行 `nvidia-smi -L`，安装支持该显卡的新版驱动后重开终端。无需为此安装 CUDA Toolkit。
- **模型不存在**：运行 `setup-model.cmd`；使用 `--profile official` 时先下载该配置的模型。手动下载必须使用本项目模型仓库中对应的文件名。
- **显存预算不足 / 速度突然很慢**：关闭其他 GPU 程序，降低 `--parallel`，检查上下文是否过大。Windows 使用共享内存后可能显著变慢。
- **译文被截断**：检查 summary，适当增加输出 token 上限和上下文，或把原文分段。
- **端口已被占用**：用 `stop-server.cmd` 停止本项目管理的服务，或 `start-server.cmd -Port 18081` 更换端口并更新客户端地址。

量化和模型本身可能产生误译。吞吐测试中的“完整输出”只表示正常结束，不表示译文语义完全正确。
