# 4 位模型与下载清单

[NVFP4](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-vLLM) 和 [INT4](https://huggingface.co/divingclone/Hy-MT2-1.8B-INT4-vLLM) 是两个独立的原始 checkpoint 仓库，各自根目录直接提供 `model.safetensors`、`config.json`、generation config、tokenizer 和 chat template。模型权重不进入程序包。

`setup-model.cmd` 按显卡选择并下载；`setup-model.cmd --profile compat` 明确选择普通 INT4。每个文件固定提交、大小和 SHA-256，支持断点续传，所有文件校验通过后才原子发布 checkpoint 目录。下载后可离线运行，不需要 Hugging Face 账号。NVFP4 checkpoint 为 1,373,024,141 字节，INT4 为 1,300,648,141 字节。

`manifest.json` 为 schema 2：`repositories.fast/compat` 固定两个仓库及提交，`checkpoint_files` 与 `checkpoints.compat.checkpoint_files` 分别固定文件哈希。`files` 中的 ZIP 信息仅保留给离线导入及旧模型缓存复用，不用于网络下载。导入示例：`setup-model.cmd --source X:/models/Hy-MT2-1.8B-NVFP4-vllm.zip`，仅接受清单对应的 ZIP。

RTX 50 自动使用 NVFP4 W4A4；RTX 30/40 使用 GPTQ INT4 W4A16，对称 group_size=128。两者 embedding/head 保留 BF16，默认 KV 均为 INT8。RTX 30/40 尚待实卡验证。

维护者使用 `scripts/publish_vllm_models.py` 发布已校准目录的原始文件，逐文件核对远程内容并固定提交后再打包程序。`scripts/build_model_bundle.py` 仍可生成可选离线 ZIP。模型遵守原始 Hy-MT2 许可证。
