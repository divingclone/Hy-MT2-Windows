# 公开基准证据

本目录只保存实际测量的精简摘要，不包含用户路径、GPU UUID、PID、提示或逐条译文。完整评测 JSON 保留在本机忽略的 `results/`，历史报告集中在 `results/public-evidence-detailed/`，现行优化矩阵在 `results/graph-coverage-official-c32/`，历史官方 API 矩阵在 `results/official-nonstream-api-async/`，旧同步客户端矩阵在 `results/official-nonstream-api/`；`detailed_report_sha256` 可核对其原始字节。版本控制保留结论、条件、聚合指标和失败计数，不收录重复的完整运行记录。

| 文件 | 内容 |
| --- | --- |
| [graph-coverage-c32.json](graph-coverage-c32.json) | 32 并发同配置 CUDA Graph 覆盖对照：64 → 256 token，三组独立进程、每进程三轮热测，完整响应 TPS 与延迟、启动及捕获分配，见 [优化说明](../docs/LOW_CONCURRENCY_TPS.md) |
| [graph-coverage-memory-c32.json](graph-coverage-memory-c32.json) | 32 并发服务进程树 WDDM 显存：请求阶段峰值增加 104 MiB，首次重编译启动峰值相同；新旧各两个进程、100 ms 目标间隔 |
| [graph-coverage-validation.json](graph-coverage-validation.json) | 图覆盖优化的固定调度逐 token 对照、256 并发运行回归、INT4 批量功能验证、运行时与权重身份 |
| [official-nonstream-api-async.json](official-nonstream-api-async.json) | 历史官方对照：双方使用 aiohttp 异步连接池的非流式翻译 API；未修改官方 llama.cpp + 腾讯 Q4_K_M 对比 vLLM，32/256 并发、三个独立进程、每次三轮热测，见 [优化报告](../docs/API_OPTIMIZATION.md) |
| [api-optimization.json](api-optimization.json) | 同进程客户端交错对照、连接复用/事件循环/调度预算消融、相同配置的核心复查，保留 Windows 双前端启动失败和排除的干扰轮次 |
| [official-nonstream-api.json](official-nonstream-api.json) | 历史同步 urllib 客户端的非流式 API 对照；包含客户端供给瓶颈，见 [历史报告](../docs/OFFICIAL_API_BENCHMARK.md) |
| [teacher-fidelity.json](teacher-fidelity.json) | 原始 BF16 输出参考的量化消融，含 32/256 并发、译文一致性、固定前缀 token 概率与重复性检查，见 [保真度报告](../docs/TEACHER_FIDELITY.md) |
| [vllm-kv-comparison.json](vllm-kv-comparison.json) | 8 位 KV 主对照：vLLM BF16 / INT8 与 llama.cpp Q8，32/256 并发三组交错共 72 轮；相同 vLLM token 容量下显存、吞吐、独立质量和其他 8 位路径，见 [KV 报告](../docs/VLLM_KV.md) |
| [vllm-core-comparison.json](vllm-core-comparison.json) | 原 16 位 KV 核心对照：两端预分词、同进程预热、32/256 并发，三组交错进程共 48 轮，含独立初始化/首轮、输出等价校验、GPU 遥测和历史原程序 A/B；核心吞吐 1.24 / 3.84 倍，见 [迁移报告](../docs/VLLM_MIGRATION.md) |
| [vllm-windows-migration.json](vllm-windows-migration.json) | vLLM 迁移补充：HTTP 实际在途峰值、冷/热轮、早期批量调优、显存与独立翻译质量；原生首轮与 vLLM 预热探索值不用于计算核心加速比 |
| [readme-kv-comparison.json](readme-kv-comparison.json) | 历史 llama.cpp 主表：官方 Q4_K_M/F16、本项目局部校准 NVFP4/F16 与 Q8_0；沿用 512 条、256 并发、1024 上下文，九次同期速度与显存重测 |
| [current-build-validation.json](current-build-validation.json) | 历史 llama.cpp 程序的五架构编译审计、102 项 CUDA 算子检查、8 项 Q8 attention 检查、364 次采样对照及 NVFP4 量化数值测试 |
| [current-quality-validation.json](current-quality-validation.json) | 历史 llama.cpp 程序与当时清单模型的 F16/Q8 混合长度质量回归；各 256 条、4096 上下文、2048 生成预算，含历史基线、chrF++/BLEU 与配对区间 |
| [nvfp4-q8-mixed-evaluation.json](nvfp4-q8-mixed-evaluation.json) | 固定 OPUS-100 中英分集校准/评测，256 条随机长短请求；27 次运行含失败、chrF++/BLEU 配对区间、Q8 融合和 attention 消融、模型及二进制哈希 |
| [nvfp4-kv-evaluation.json](nvfp4-kv-evaluation.json) | 新增离线 MSE/imatrix 和 Q8/Q4 KV：24 次 GPU 批量运行、显存采样、4 条独立样例的 BF16 分布对照；不是正式翻译质量评分 |
| [upstream-comparison.json](upstream-comparison.json) | 历史主对照：未修改原版 Q4_K_M → 本项目 NVFP4；附加对照：两端 NVFP4。各三对，共 12 次运行，包含参数、计数、哈希和有效性复核 |
| [upstream-release.json](upstream-release.json) | 官方发布资产、提交关系、二进制未修改证明、编译器及运行库差异 |
| [rtx5090-windows.json](rtx5090-windows.json) | 13 组、27 次历史运行，包含优化阶段、后续波动和单次消融 |
| [rtx5090-vram.json](rtx5090-vram.json) | RTX 5090 的 3 个批处理显存观测和 1 个已加载服务观测 |
| [source-reproducibility.json](source-reproducibility.json) | 固定提交、完整源码补丁和恢复后源码树校验 |

口径和比较边界见 [性能报告](../docs/PERFORMANCE.md)，输入集见 [benchmark_cases.json](../scripts/benchmark_cases.json)。主结果将量化、源码、调度、接口和构建差异计入整体方案收益，不称作纯 kernel 加速。

`source_report_sha256` 标识脱敏来源；完整本机报告和运行历史不随精简仓库发布。`output_jsonl_sha256` 标识实际输出文件，HTTP JSONL 还包含时延等字段，因此不是跨接口译文相同的证明。主对照不宣称两种量化输出一致或质量等效。

README 历史主表的 `completion_tokens_per_second` 和历史原版对照的 `output_tokens_per_second` 均使用 completion 口径，包含真实 EOG，不计填充 token。旧原生 `wall_s` 包含首次图准备和读写，排除模型初始化；HTTP 墙钟包含队列与请求，但不含后续 JSONL 磁盘写出。新 vLLM 核心主对照提前准备提示 token ID，排除输入/输出处理，原生使用新增的 `core_wall_s`，两端首次轮和初始化都单列。原版含启动计时截止于响应完成，本项目进程总时间包含退出，两者不能直接当成相同口径。

当前非流式 API 对照使用 `scripts/prepare_official_baseline.py`、`scripts/benchmark_official_api.py` 与 `scripts/benchmark_backend.py`，均保留在仓库；SSE 需显式传入 `--stream`。公开摘要保留测量条件、逐次指标、模型与程序身份及失败记录；详细结果保留在忽略的 `results/`。旧 llama.cpp 的测量工具归档在 `archive/llama-cpp/scripts/`，早期临时工具保留于本机 `.local/validation-tools/`。

历史记录保留较慢的复测、单次便携验证和单次源码消融，不能混成同一组重复测量或只取最高值。只有 RTX 5090 实测；显存建议不是其他显卡的峰值保证。
