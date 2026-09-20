# vLLM 便携发布

运行 Python/前端/Rust 测试和真实 GPU 鉴权、流式、重启、进程树清理测试，校验模型清单，再运行 `scripts/build_desktop.py`。当前大体积 vLLM 运行时使用 Zip64 便携包；旧 NSIS 小运行时流程不再用于本版本。

签名仍只使用用户已有的 `TAURI_SIGNING_PRIVATE_KEY` 和 `HYMT_UPDATER_PUBLIC_KEY`；构建脚本不生成私钥。`--signed` 只签名便携 ZIP 并生成对应平台的更新描述。没有密钥时生成本地未签名包，不自动发布。

发布前核对托管平台单文件容量限制，必要时设计独立运行时分发；不要把大于平台限制的本地 ZIP 当作已可下载 Release。NVFP4 与 INT4 的原始 checkpoint 文件分别托管于两个 Hugging Face 仓库，默认程序包不包含权重。先用 `scripts/publish_vllm_models.py` 上传模型并校验匿名访问、远程大小和 SHA-256，将项目清单固定到该提交后再构建程序包。用户在模型页下载一个对应显卡的模型，或离线导入 ZIP。

不要随包包含缓存、实验环境、原始 API Key、旧 GGUF 或本机详细运行记录。模型独立发布不代表已发布 GitHub 程序 Release；旧线上程序 Release 保持不变。
