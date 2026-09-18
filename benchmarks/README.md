# 公开基准证据

本目录只保存实际测量的精简摘要，不包含用户路径、GPU UUID、PID、提示或逐条译文。完整评测 JSON 已移至本机忽略的 `results/public-evidence-detailed/`；`detailed_report_sha256` 可核对其原始字节。版本控制保留结论、条件、聚合指标和失败计数，不收录重复的完整运行记录。

| 文件 | 内容 |
| --- | --- |
| [readme-kv-comparison.json](readme-kv-comparison.json) | README 当前主表：官方 Q4_K_M/F16、本项目局部校准 NVFP4/F16 与 Q8_0；沿用 512 条、256 并发、1024 上下文，九次同期速度与显存重测 |
| [current-build-validation.json](current-build-validation.json) | 当前程序的五架构编译审计、102 项 CUDA 算子检查、8 项 Q8 attention 检查、364 次采样对照及 NVFP4 量化数值测试 |
| [current-quality-validation.json](current-quality-validation.json) | 当前程序与清单模型的 F16/Q8 混合长度质量回归；各 256 条、4096 上下文、2048 生成预算，含历史基线、chrF++/BLEU 与配对区间 |
| [nvfp4-q8-mixed-evaluation.json](nvfp4-q8-mixed-evaluation.json) | 固定 OPUS-100 中英分集校准/评测，256 条随机长短请求；27 次运行含失败、chrF++/BLEU 配对区间、Q8 融合和 attention 消融、模型及二进制哈希 |
| [nvfp4-kv-evaluation.json](nvfp4-kv-evaluation.json) | 新增离线 MSE/imatrix 和 Q8/Q4 KV：24 次 GPU 批量运行、显存采样、4 条独立样例的 BF16 分布对照；不是正式翻译质量评分 |
| [upstream-comparison.json](upstream-comparison.json) | 历史主对照：未修改原版 Q4_K_M → 本项目 NVFP4；附加对照：两端 NVFP4。各三对，共 12 次运行，包含参数、计数、哈希和有效性复核 |
| [upstream-release.json](upstream-release.json) | 官方发布资产、提交关系、二进制未修改证明、编译器及运行库差异 |
| [rtx5090-windows.json](rtx5090-windows.json) | 13 组、27 次历史运行，包含优化阶段、后续波动和单次消融 |
| [rtx5090-vram.json](rtx5090-vram.json) | RTX 5090 的 3 个批处理显存观测和 1 个已加载服务观测 |
| [source-reproducibility.json](source-reproducibility.json) | 固定提交、完整源码补丁和恢复后源码树校验 |

口径和比较边界见 [性能报告](../docs/PERFORMANCE.md)，输入集见 [benchmark_cases.json](../scripts/benchmark_cases.json)。主结果将量化、源码、调度、接口和构建差异计入整体方案收益，不称作纯 kernel 加速。

`source_report_sha256` 标识脱敏来源；完整本机报告和运行历史不随精简仓库发布。`output_jsonl_sha256` 标识实际输出文件，HTTP JSONL 还包含时延等字段，因此不是跨接口译文相同的证明。主对照不宣称两种量化输出一致或质量等效。

当前主表的 `completion_tokens_per_second` 和历史原版对照的 `output_tokens_per_second` 均使用 completion 口径，包含真实 EOG，不计填充 token。原生 `wall_s` 包含首次图准备和读写，排除模型初始化；HTTP 墙钟包含队列与请求，但不含后续 JSONL 磁盘写出。原版含启动计时截止于响应完成，本项目进程总时间包含退出，两者不能直接当成相同口径。

当前主表的临时测量、复核和评分工具保留在本机 `.local/validation-tools/`，不加入 Git。公开摘要保留测量条件、逐次指标、模型与程序身份及失败记录；详细结果保留在忽略的 `results/`。历史官方对照仍可用仓库现有 `benchmark_upstream.py` 与 `summarize_upstream_benchmark.py` 复现。

历史记录保留较慢的复测、单次便携验证和单次源码消融，不能混成同一组重复测量或只取最高值。只有 RTX 5090 实测；显存建议不是其他显卡的峰值保证。
