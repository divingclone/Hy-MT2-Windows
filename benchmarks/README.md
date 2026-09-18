# 公开基准证据

本目录只保存实际测量的白名单摘要，不包含用户路径、GPU UUID、PID、提示或逐条译文。

| 文件 | 内容 |
| --- | --- |
| [upstream-comparison.json](upstream-comparison.json) | 主对照：未修改原版 Q4_K_M → 本项目 NVFP4；附加对照：两端 NVFP4。各三对，共 12 次运行，包含参数、计数、哈希和有效性复核 |
| [upstream-release.json](upstream-release.json) | 官方发布资产、提交关系、二进制未修改证明、编译器及运行库差异 |
| [rtx5090-windows.json](rtx5090-windows.json) | 13 组、27 次历史运行，包含优化阶段、后续波动和单次消融 |
| [rtx5090-vram.json](rtx5090-vram.json) | RTX 5090 的 3 个批处理显存观测和 1 个已加载服务观测 |
| [source-reproducibility.json](source-reproducibility.json) | 固定提交、完整源码补丁和恢复后源码树校验 |

口径和比较边界见 [性能报告](../docs/PERFORMANCE.md)，输入集见 [benchmark_cases.json](../scripts/benchmark_cases.json)。主结果将量化、源码、调度、接口和构建差异计入整体方案收益，不称作纯 kernel 加速。

`source_report_sha256` 标识脱敏来源；完整本机报告和运行历史不随精简仓库发布。`output_jsonl_sha256` 标识实际输出文件，HTTP JSONL 还包含时延等字段，因此不是跨接口译文相同的证明。主对照不宣称两种量化输出一致或质量等效。

`output_tokens_per_second` 使用 completion 口径，包含真实 EOG，不计填充 token。原生 `wall_s` 包含首次图准备和读写，排除模型初始化；HTTP 墙钟包含队列与请求，但不含后续 JSONL 磁盘写出。原版 `loaded_process_wall_s` 截止于响应完成，本项目进程总时间包含退出，两者不能直接当成相同口径。

主结果由 [summarize_upstream_benchmark.py](../scripts/summarize_upstream_benchmark.py) 生成，检查请求数、逐条 token 合计、结束状态、独立种子与实际文件哈希；原版发布证明和逆重排片段证明同时收录。较早 NVFP4 对照未在运行前记录优化二进制哈希，摘要另保留聚合时的文件哈希并标明时间边界；主 Q4_K_M→NVFP4 对照已在运行前记录两端哈希。

历史记录保留较慢的复测、单次便携验证和单次源码消融，不能混成同一组重复测量或只取最高值。只有 RTX 5090 实测；显存建议不是其他显卡的峰值保证。
