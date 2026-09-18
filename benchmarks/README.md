# 公开基准证据

本目录只保存从实际测量报告按字段白名单提取的摘要，不包含用户路径、GPU UUID、PID、提示或逐条译文。

| 文件 | 内容 |
| --- | --- |
| [rtx5090-windows.json](rtx5090-windows.json) | 13 组、27 次关键运行；逐次吞吐、时间、计数、配置、报告及输出 SHA-256 |
| [rtx5090-vram.json](rtx5090-vram.json) | RTX 5090 的 3 个批处理显存观测和 1 个已加载服务观测 |

口径和比较边界见 [性能报告](../docs/PERFORMANCE.md)，输入集见 [benchmark_cases.json](../scripts/benchmark_cases.json)。

`source_report` 是原始本机报告的文件名，`source_report_sha256` 用于标识提取来源；原报告和完整运行历史不随精简仓库发布。摘要不能替代在目标显卡上复测。`output_jsonl_sha256` 记录当次完整输出文件的哈希，用于核对已报告的一致性，不包含译文内容。

`output_tokens_per_second` 使用 completion 口径，包含真实 EOG，不计填充 token。`wall_s` 排除原生模型初始化但包含首次图准备；`process_wall_s_including_initialization` 包含初始化。HTTP 的计时边界不同。详细定义同时写在 JSON 的 `metric_definitions` 中。

后续较慢的复测、单次便携验证和单次源码消融均保留，不能把它们混成同一组重复测量，也不能仅选最高值代表持续性能。只有 RTX 5090 实测；显存推荐不是对其他显卡的峰值保证。
