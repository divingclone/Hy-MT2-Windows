# llama.cpp 历史归档

此目录保留切换到 vLLM 前的启动器、GGUF 量化/源码工具、测试及文档，供追溯历史实现和实验。不会进入当前运行包，不再用于桌面启动或 CI 主检查。

`huggingface/` 保留旧 GGUF 模型卡模板及来源记录；模板中的清单和许可证链接按当时 Hugging Face 仓库布局填写。当前 safetensors 模型卡位于根目录的 `huggingface/vllm/`。

历史脚本使用原目录布局及旧依赖；归档后不能直接当成现行入口运行。需要重现旧引擎时，使用旧提交或将历史布局恢复到独立目录，按旧清单重新准备源码/模型。完整实测记录和模型哈希保留于项目 `benchmarks/` 与本机 `results/`。当前 vLLM 入口见仓库根 README。

本机保留旧源码于 `src/`、原生产二进制于 `bin/`，以及单个 NVFP4 基准模型于 `models/`（均不进 Git）。重复旧安装包和量化候选已清理，清理清单位于 `results/vllm-adoption-cleanup.json`。原始脚本是历史快照，重现时须调整根路径和依赖路径。
