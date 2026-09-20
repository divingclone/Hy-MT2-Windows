# vLLM 便携发布

0.2.3 使用已验收的定制 PyTorch payload，构建流程见 [定制后端](PYTORCH_CUSTOM_BUILD.md)。发布新版本时同步 package.json/package-lock.json、Cargo.toml/Cargo.lock 和 tauri.conf.json，重建桌面 EXE，再用 `scripts/pytorch/package_backend.py` 打包该 payload；不要重新从完整开发运行时 staging 覆盖定制后端。签名、更新清单和 Release 标签均须指向同一版本与提交。该版本沿用已有 GPU 验证记录，仅做必要的构建、签名、归档和发布校验，不重复 GPU 请求矩阵；验证限制须写入发布说明。

运行 Python/前端/Rust 测试和真实 GPU 鉴权、流式、重启、进程树清理测试，校验模型清单，再运行 `scripts/build_desktop.py`。当前大体积 vLLM 运行时使用 Zip64 便携包；旧 NSIS 小运行时流程不再用于本版本。

签名仍只使用用户已有的 `TAURI_SIGNING_PRIVATE_KEY` 和 `HYMT_UPDATER_PUBLIC_KEY`；构建脚本不生成私钥。`--signed` 只签名便携 ZIP 并生成对应平台的更新描述。没有密钥时生成本地未签名包，不自动发布。

发布前核对托管平台单文件容量限制，必要时设计独立运行时分发；不要把大于平台限制的本地 ZIP 当作已可下载 Release。NVFP4 与 INT4 的原始 checkpoint 文件分别托管于两个 Hugging Face 仓库，默认程序包不包含权重。先用 `scripts/publish_vllm_models.py` 上传模型并校验匿名访问、远程大小和 SHA-256，将项目清单固定到该提交后再构建程序包。用户在模型页下载一个对应显卡的模型，或离线导入 ZIP。

默认运行时使用 NVIDIA 的 `GRAPH_JIT_ONLY` cuDNN 配置，只保留 dispatcher、graph 和 runtime-compiled 三个子库。普通 ZIP 继续使用原有 Deflate 级别 5；不采用分卷或自解压格式。构建后核对大小小于 2,000,000,000 字节，并在发布前运行空缓存、NVFP4/INT4、32/256 并发裁剪回归。详细记录见 [运行包精简](RUNTIME_SIZE.md)。

不要随包包含缓存、实验环境、原始 API Key、旧 GGUF 或本机详细运行记录。模型独立发布不代表已发布 GitHub 程序 Release；旧线上程序 Release 保持不变。
