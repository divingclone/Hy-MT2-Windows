# NVFP4 校准与 Q8 缓存

当前默认 NVFP4 模型使用 OPUS-100 中英翻译语料校准和局部加权 MSE 尺度搜索。桌面端默认 Q8 KV；原生批量命令行保留 F16 默认值，可显式选择 Q8_0。模型来源、哈希与校准参数见 [provenance.json](../huggingface/provenance.json)，测量结果见 [性能报告](PERFORMANCE.md)。

Windows 原生 vLLM 的独立启动入口默认使用动态 INT8 KV，权重仍为 NVFP4；其格式与 llama.cpp Q8_0 不同。32/256 并发的同口径吞吐、显存和质量对照见 [vLLM 8 位 KV 测试](VLLM_KV.md)。

## 制作模型

从原始、未融合的 BF16/F16/F32 模型量化，不从已有低位模型反量化。按 [构建说明](BUILD.md) 编译并安装 `llama-quantize`、`llama-imatrix` 到 `bin`，或通过 `--binary-dir` 指定构建目录。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_opus_calibration.py --output-dir results/opus-calibration
.\.venv\Scripts\python.exe scripts/quantize_nvfp4.py --source models/Hy-MT2-1.8B-BF16.gguf --output models/Hy-MT2-1.8B-NVFP4-calibrated-fused.gguf --calibration results/opus-calibration/calibration.txt --parse-special --context 2048 --chunks 64 --scale-search local
```

输出必须使用尚不存在的文件名。脚本验证高精度输入、imatrix 结构和工具选项，先量化再重排，并记录输入、校准、程序和模型哈希。已有同一高精度模型的 imatrix 可通过 `--imatrix` 传入；`--dry-run` 只检查输入和命令。

局部搜索在参考 FP8 尺度附近选择加权误差更小的候选，保留参考值，因此块内目标误差不会增加。它只改变离线权重制作，不改变 NVFP4 块大小和推理形状。`--scale-search full` 保留作全尺度离线实验；本次独立翻译评分支持局部搜索作为默认，不能把权重误差最小等同于译文最好。

OPUS 下载固定版本和哈希，validation 用于校准、test 用于评估，并排除跨集重复。长度随机混合，不重复句子填充长度，不按参考译文缩短生成预算。这个子集不是生产流量，也不是完整官方 OPUS 分数。块级数值回归保存在源码 `test-quantize-fns`；参考译文评分应独立于校准数据。

## Q8 缓存

```powershell
.\translate-batch.cmd input.jsonl output.jsonl --cache-type-k q8_0 --cache-type-v q8_0
.\start-server.cmd -CacheTypeK q8_0 -CacheTypeV q8_0
```

Q8_0 是包含块尺度的 GGML 整数量化，不是 FP8。融合内核在 RoPE、RMSNorm 和缩放后直接量化写入 K 缓存，减少中间读写与独立启动；保持与原 Q8 写入相同的舍入规则，不依赖固定请求长度。

每个上下文 token 的 K/V 合计空间：F16 为 64 KiB，Q8_0 为 34 KiB。每序列 1024 token 对应 64/34 MiB，Q8 缓存理论减少 46.875%；整进程还包含权重和工作区，实测显存及吞吐取舍以 README 表格为准。混合 K/V 类型可能发生转换，默认建议两者一致。

默认启用 Q8 写入融合，实验 attention subwarp 保持关闭。对新语言、领域及长尾输入，仍需检查完整结束、数字、术语、格式与漏译，不能仅依据固定长度跑分。

## 与 Unsloth 的关系

[Unsloth Dynamic NVFP4](https://unsloth.ai/docs/basics/nvfp4) 的校准和敏感层思路可参考，但 checkpoint、W4A4 内核及 KV 格式不能直接套到本仓库。当前 Blackwell MMQ 路径已有动态 FP4 激活量化，其他计算路径可能不同。带额外投影尺度的 checkpoint 会被重排脚本拒绝，不能直接替换本工程模型。

本次临时跑分、评分、压力测试与详细报告保存在忽略的 `.local/validation-tools/` 和 `results/`；Git 只保留量化制作工具、核心回归及精简结果。
