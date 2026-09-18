# 模型目录

运行环境包中的 `setup-model.cmd` 会根据显卡下载适合的模型；下载地址与 SHA-256 在 [manifest.json](manifest.json) 中。

- RTX 50（CC 12.0）：`Hy-MT2-1.8B-NVFP4-fused.gguf`
- GTX 16、RTX 20/30/40：`Hy-MT2-1.8B-Q4_K_M-fused.gguf`

模型文件在 [Hugging Face](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF) 分发，不进入 Git。手动下载时保持文件名不变。

这些是本项目专用的重排 GGUF，使用修改后的推理程序。模型许可为 Apache-2.0，来源与改动见 [模型说明](../huggingface/README.md)。
