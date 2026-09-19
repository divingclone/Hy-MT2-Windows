# 发布到 GitHub 与 Hugging Face

Git 保存代码、补丁和文档；GitHub Releases 保存桌面安装包、免安装 ZIP 和命令行运行环境 ZIP；Hugging Face 保存两个 GGUF。不要把模型、DLL、SDK、个人 Token 或测试历史加入 Git。

## 维护者环境

执行 `setup-dev.cmd`（需要 CPython 3.12；已有本地开发环境可直接使用）。运行用户无需这个步骤。开发依赖保存在 `.venv`，发布环境不包含 pip 或 Hugging Face SDK。

## 上传模型

公开模型仓库为 [`divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF`](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF)。

1. 在 [Hugging Face 设置](https://huggingface.co/settings/tokens) 创建有目标仓库写权限的 Token。
2. 双击 `huggingface-login.cmd`，在本机隐藏输入框粘贴 Token。不要在聊天、命令参数或 Git 文件中写 Token。
3. 执行上传：

```powershell
.\.venv\Scripts\python.exe scripts/publish_models.py --status
.\.venv\Scripts\python.exe scripts/publish_models.py --upload
```

脚本只上传 `models/manifest.json` 指定的两个模型以及模型卡、改动说明、许可证、校验清单和来源记录；上传前检查本地哈希，上传后检查 Hub 上的大小与 LFS SHA-256。登录与上传结果保存在被 Git 忽略的 `.local/huggingface` 和 `.local/huggingface-upload.json`。不会上传整个工作目录。

用户首次下载由 `setup-model.cmd` 完成，使用标准库 HTTPS、断点续传、文件大小和 SHA-256 检查。模型清单可将 `revision` 固定到上传后的 commit ID；即使使用 `main`，文件仍必须匹配记录的 SHA-256。

## 仅更新模型卡

发布代码、性能结果或下载链接后，更新本地 `huggingface/README.md` 和 `huggingface/MODEL_CHANGES.md`，再执行：

```powershell
.\.venv\Scripts\python.exe scripts/publish_models.py --update-card
```

此模式只向已经存在的模型仓库提交 `README.md` 和 `MODEL_CHANGES.md`，不创建仓库，不读取或上传本地 GGUF，也不修改模型、manifest、许可证或来源记录。提交前后均检查仓库可见性、两个远程模型的大小与 LFS SHA-256 是否匹配 `models/manifest.json`；使用提交前的 `main` commit ID 防止覆盖并发更新，并确认其他文件未变。默认要求公开仓库；私有仓库必须显式传 `--private`。

独立结果写入 `.local/huggingface-card-update.json`，保留首次上传证据 `.local/huggingface-upload.json`。`--update-card` 与 `--upload`、`--login`、`--status` 互斥；继续使用已有本地登录，不把 Token 写进命令。上传后应匿名打开模型页面，确认 GitHub 和 Release 链接可访问。

## 打包运行环境

源码构建完成并通过 GPU 回归后，将程序、项目 DLL 和 `build-info.json` 一起更新到 `bin`。启动器与打包器均使用这个目录；模型以 `models/manifest.json` 为准，NVFP4 为 OPUS 局部校准权重。构建目录只用于编译，不是另一套用户运行入口。

```powershell
.\scripts\run-python.cmd scripts/package_windows.py --bin-dir bin --dry-run
.\scripts\run-python.cmd scripts/package_windows.py --bin-dir bin
```

默认产生 `dist/HyMT-Windows-NVIDIA-runtime.zip`，包含 Windows x64 程序、独立 Python、CUDA/MSVC 运行库、模型下载清单与启动脚本，不包含模型。脚本白名单复制、审计 DLL 依赖、校验 SHA-256 并检查 ZIP，拒绝覆盖已有输出。重新打包请传新的 `--output dist/名称`。

需要离线大包时加 `--include-models`；该包可能超过 GitHub 单附件限制，仅适合其他分发渠道。GitHub 要求每个 Release 附件小于 2 GiB，默认运行包会强制检查这一点。[GitHub 官方说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

完成后验证真实下载、搬迁和 GPU 推理：

```powershell
.\scripts\run-python.cmd scripts/validate_portable.py
```

验证器将 ZIP 解压到新的中文/空格目录，隔离开发环境，校验模型并检查批量翻译与 API 启停。`--model-directory models` 可使用本机模型，但不能据此声称 Hub 下载链路已验证。详细跑分、评分和持续负载工具仅保留在本机 `.local/validation-tools/`，不随源码和运行包分发；公开结果保留在 `benchmarks` 的精简摘要中。

## 上传 GitHub

公开代码仓库为 [`divingclone/Hy-MT2-Windows`](https://github.com/divingclone/Hy-MT2-Windows)，当前运行环境发布到 [`v0.2.0` Release](https://github.com/divingclone/Hy-MT2-Windows/releases/tag/v0.2.0)。维护者在项目根目录确认 `origin` 指向该仓库后推送：

```powershell
git remote get-url origin
# 应为 https://github.com/divingclone/Hy-MT2-Windows.git
git push -u origin main
```

在该仓库的 [Releases 页面](https://github.com/divingclone/Hy-MT2-Windows/releases) 为对应源码提交创建版本标签和 Release，附加 `HyMT-Windows-NVIDIA-runtime.zip` 和对应 `.sha256` 文件。本次版本为 `v0.2.0`。权重变更时执行上面的 `--upload`，仅说明变更时执行 `--update-card`，同步 Hugging Face。运行用户下载运行环境包，不能把 GitHub 自动生成的 Source code ZIP 当作运行环境。

上传代码前确认 `git status` 不包含模型、二进制或 `.local`，并使用自己的提交署名。CI 在 Windows 上运行无 GPU 单元测试及固定源码补丁复现；真实 GPU 测试仍需本地进行。


## 桌面端正式发布

桌面端当前版本为 `desktop-v0.1.3`，与命令行运行核心的 `v0.2.0` 使用独立版本号。构建、更新签名与模型目录规则见 [桌面端说明](DESKTOP.md)。

一次完整发布应包含：

- `HyMT_0.1.3_x64-setup.exe` 与 `.exe.sig`。
- `HyMT-0.1.3-windows-x64-portable.zip` 与 `.zip.sig`。
- `desktop-latest.json`，其版本、下载 URL 和签名须与附件一致。
- `HyMT-Windows-NVIDIA-runtime.zip` 与 `.zip.sha256`，保留命令行用户的固定下载入口。
- `SHA256SUMS.txt` 与 `RELEASE-NOTES.md`。

先推送经过检查的源码与标签，创建草稿发布并上传全部附件。核对附件大小、SHA-256 与签名，等待 Windows 和桌面端 CI 通过，再将发布设为正式版和 Latest。公开后匿名验证下载与 `releases/latest/download/desktop-latest.json`，确保应用更新入口可访问。发布目录只保留交付文件，模型、用户配置、密钥、编译目录和本机测试记录不上传。
