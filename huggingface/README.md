# Hugging Face 模型发布文件

当前 vLLM 模型卡模板与修改说明位于 [vllm/](vllm/README.md)，由 `scripts/publish_vllm_models.py` 发布到 NVFP4 和 INT4 两个独立仓库。权重、配置和 tokenizer 以原始文件分发，固定版本及哈希记录于 [模型清单](../models/manifest.json)。

维护者使用 `huggingface-login.cmd` 在本机登录。登录只保存本地凭据；发布模型需另行执行发布脚本。用户下载公开模型无需登录。

旧 GGUF 模型卡和来源记录已移至 [历史归档](../archive/llama-cpp/README.md)。
