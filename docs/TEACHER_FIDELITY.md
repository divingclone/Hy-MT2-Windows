> 本文是迁移前的实验记录，保留原测试条件和命令。项目现已采用 vLLM；现行运行入口和默认配置见 [README](../README.md)。旧原生引擎工具已归档，不随当前运行包提供。

# 相对未量化 BF16 模型的量化保真度

本评测用原始 Hy-MT2-1.8B BF16 权重、BF16 KV 的输出作为唯一参考；OPUS 数据集只提供输入文本，不使用其中的译文评分。目标是定位量化与推理策略导致的偏离，不能证明参考模型本身翻译正确，也不能把文本一致性分数解释成“翻译质量保留百分比”。

## 条件与控制

- Windows 原生，RTX 5090；沿用已有 BF16 源权重作为参考，没有改变部署使用 4 位权重的目标。
- 独立中英输入 256 条：两个目标语言各 128 条；短/中/长为 128/64/64，与量化校准输入分离。
- 所有组固定同一提示、temperature=0、top_p=1、top_k=1、repetition_penalty=1.05。上下文 4096，最大生成 2048，不截短输入或参考译文。
- vLLM 各组统一 TRITON_ATTN，避免将 FlashAttention/Triton 切换混入 KV 消融。4 位组使用同一份 NVFP4 checkpoint；Marlin 为 W4A16，CUTLASS 为 W4A4，激活精度比较也包含线性内核实现差异。
- llama.cpp 使用当前 NVFP4 GGUF 与现有 DLL。其量化校准、embedding 精度及内核不同，跨引擎差距属于整套方案差距。
- 32 并发完成策略消融；256 并发复核原始模型与两种主要 CUTLASS KV 方案。每个配置首轮和复测各一次，检查输出重复性，不将这两次质量运行替代此前三进程性能基准。
- vLLM 的 BF16 KV 池为 32 并发 3 GiB、256 并发 8 GiB；INT8 按相同 token 容量乘 132/256。256 的预算针对本次较长质量输入增加，与原 1024 上下文性能测试分开。

## 结果：32 并发

主值取预热后输出。表中所有文本分数都相对同一份原始 BF16 译文，100 表示一致；不是“质量保留率”。W4A16/W4A4 的权重均为 4 位，区别在激活计算及内核。

| 策略 | 译文 chrF++ | 完全一致 / 256 | 固定前缀 top-1 与 BF16 一致 | 参考 token NLL |
| --- | ---: | ---: | ---: | ---: |
| 原始 BF16 权重 + BF16 KV | 100.0000 | 256 | 100.00% | 0.138984 |
| 原始 BF16 权重 + INT8 KV | 97.1826 | 166 | 99.26% | 0.139481 |
| NVFP4 W4A16 + BF16 KV，Marlin | 84.0585 | 46 | 92.57% | 0.243760 |
| NVFP4 W4A4 + BF16 KV，CUTLASS | 79.2287 | 30 | 89.46% | 0.347265 |
| NVFP4 W4A4 + INT8 KV，CUTLASS | 79.5983 | 29 | 89.34% | 0.348429 |
| 当前 llama.cpp NVFP4 + F16 KV | 80.7150 | 29 | 未测 | 未测 |
| 当前 llama.cpp NVFP4 + Q8 KV | 80.4338 | 32 | 未测 | 未测 |

主要偏离来自 **4 位权重及 W4A4 激活路径**，8 位 KV 的额外影响较小：

| 变化 | NLL 增量（nats/token） | 95% 配对区间 |
| --- | ---: | --- |
| BF16 → W4A16，保留 BF16 KV | +0.104776 | [0.097267, 0.111577] |
| W4A16 → W4A4，保留 BF16 KV | +0.103506 | [0.095956, 0.111293] |
| 原始权重，仅 BF16 KV → INT8 KV | +0.000497 | [0.000174, 0.000826] |
| W4A4，仅 BF16 KV → INT8 KV | +0.001164 | [−0.007971, 0.010333] |

W4A16 相对 W4A4 的文本保真度也更高，差距约 4.83 chrF++；这一比较同时包含 Marlin/CUTLASS 实现差异，不能称为纯激活量化误差。W4A4 的 INT8 KV 文本分数高 0.37，区间 [−0.36, 1.15]，不能据此宣称 KV 量化提高质量。llama.cpp 的 Q8 相对 F16 差 −0.28，区间 [−1.18, 0.57]。

vLLM W4A4/INT8 相对当前 llama.cpp/Q8 为 −0.84 chrF++，区间 [−1.77, 0.15]；本子集没有确立谁在文本保真度上更优，也没有证明语义质量等价。结合此前性能测试，INT8 KV 仍是合理的节省缓存方案；若更重视接近原始模型的行为，已测 W4A16/BF16 KV 比 W4A4 更合适。W4A16/INT8 组合不在本次矩阵内，不将结论直接外推到该组合。

## 256 并发与重复性

| 256 并发策略 | 对固定 32 并发 BF16 参考的 chrF++ | 对 256 并发 BF16 参考的 chrF++ |
| --- | ---: | ---: |
| 原始 BF16 + BF16 KV | 98.2825 | 100.0000 |
| W4A4 + BF16 KV | 79.4851 | 79.3050 |
| W4A4 + INT8 KV | 79.1727 | 79.0487 |

256 并发下两种 W4A4 KV 策略依然接近。以固定 32 并发参考计算，INT8 − BF16 为 −0.31，区间 [−1.12, 0.55]。原始模型自身在 32/256 并发间仅 181/256 条完全相同，说明改变批处理也会改变最终措辞。固定前缀的原始概率文件在相应 32/256 配置间逐字节相同；这项 prefill 诊断没有重现自由生成的路径分叉。

主矩阵中，W4A16/BF16 的首轮与预热 chrF++ 分别为 83.9706/84.0585，184/256 条相同；W4A4/BF16 的 32 并发首轮与预热为 79.9410/79.2287，59/256 条相同。其余八种配置的两轮译文全部一致。没有将首轮混入主值，也未删除首轮证据。

针对 W4A4/BF16 另启进程复测首轮及三次预热：四轮 256 条译文均与主矩阵预热结果完全一致，分数均为 79.2287；参考 token 的原始概率差最大值为 **0**。首次差异未在这四轮重现，具体原因没有进一步归因；不能将其笼统描述为持续随机漂移。配对 bootstrap 只反映输入样本不确定性，不包含所有启动和调度差异。

合计 **11 个推理进程、24 轮自由生成、6,144 条结果**，全部正常结束，无空输出或截断；另有 9 轮、每轮 28,877 个参考 token 的概率诊断。所有主矩阵配置的提示 token ID 完全一致，共 34,363 个。测试结束后已恢复原服务参数并检查健康状态。

公开聚合数据、模型/程序/输出哈希、分组与区间见 [teacher-fidelity.json](../benchmarks/teacher-fidelity.json)。此前 32/256 并发的核心吞吐结论仍见 [KV 性能报告](VLLM_KV.md)，不使用本次质量输入的单次耗时替换性能基准。

## 如何评分

**译文层**：相对原始模型 32 并发预热输出的 corpus chrF++、整条完全一致数、字符编辑距离、长度比；分目标语言和输入长度报告。仅去除译文首尾空白。

**预测层**：将同一组“输入提示 + 原始模型完整译文 token”送入每种 vLLM 配置，以 `prompt_logprobs=1` 取得逐位置原始概率。统计参考 token 的平均负对数概率 NLL（越低越好）、相对 BF16 的 NLL 增量（越接近零越好），以及各位置原始 top-1 预测与 BF16 的一致率。包含真实 EOS，共同前缀消除了前面某个措辞变化引起的后续路径分叉。接口语义见 [vLLM SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)。

这是 teacher-forced prefill 诊断，不能替代实际逐 token decode 行为；完整自由生成另外测量。已核对本地 0.29.0 Triton decoder attention 源码：prefill 也从 KV cache 读取 K/V，逐 token/head 的量化及反量化参与该诊断。原始概率不施加自由生成的 1.05 重复惩罚，因此不要求参考生成 token 在原始概率中始终排名第一。

NLL 仅检查参考路径上的概率，top-1 仅检查最大概率 token；二者不是全词表 KL 散度，也不是语义质量裁判。较低 NLL 表示更倾向参考译文，不能据此宣称模型翻译能力更强。

实际样例 `opus-test-0048` 的输入是 “Hanging around.”，BF16 输出“闲逛着。”，W4A4/BF16 KV 输出“闲逛。”；单句 chrF++ 只有 28.68，但主要差别是措辞。这是从短句低分样例中选出的说明例，不是随机抽样的人工质量统计。更换参考来源后，仍须避免把字面差异自动判为误译。

差值区间按目标语言分层，对 256 条请求做 1000 次配对 bootstrap。NLL 按 token 数加权；不是将每个相关 token 当独立样本。区间为探索性的 95% 百分位区间，未做多重比较校正。

## 复现

先保存并停止其他 GPU 推理服务，完成后恢复原配置。矩阵脚本本身不管理生产服务。需要迁移报告中的原生 Windows 环境和已构建的核心计时程序。

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-fidelity.txt
. scripts/vllm_windows_env.ps1
.venv/Scripts/python.exe scripts/benchmark_teacher_fidelity.py --output results/my-teacher-fidelity --input results/opus-v2/evaluation.jsonl --native .local/core-benchmark-v3/hy-batch-core.exe
.venv/Scripts/python.exe scripts/score_teacher_fidelity.py --input results/my-teacher-fidelity --output results/my-teacher-fidelity/score.json
```

可用 `benchmark_vllm_offline.py` 复测指定配置，再将其目录传给评分器的 `--repeat-check`。本轮具体复测为：

```powershell
.local/vllm-win/Scripts/python.exe scripts/benchmark_vllm_offline.py --config results/my-teacher-fidelity/w4a4-bf16-p32.config.json --input results/opus-v2/evaluation.jsonl --output results/my-teacher-fidelity/repeat-check --pretokenized --greedy --save-token-ids --teacher-output results/my-teacher-fidelity/teacher-p32/warm-1.jsonl --repeats 3 --max-tokens 2048
.venv/Scripts/python.exe scripts/score_teacher_fidelity.py --input results/my-teacher-fidelity --repeat-check results/my-teacher-fidelity/repeat-check --output results/my-teacher-fidelity/score-final.json
```

原始输出、完整 token 概率、日志及进程状态保留于本机 `results/teacher-fidelity/`。数据集原始译文不会读入评分计算。
