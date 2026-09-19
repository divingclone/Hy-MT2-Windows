# HyMT Desktop 0.1.3

Windows 本地 LLM 翻译后端的首个桌面正式版。使用 Tauri 2 + Svelte 管理模型与推理服务，为其他应用提供 OpenAI 兼容 API，不包含文本翻译界面。命令行运行核心沿用 v0.2.0 的推理程序和模型。

## 下载与使用

| 附件 | 用途 |
| --- | --- |
| `HyMT_0.1.3_x64-setup.exe` | 安装版，缺少 WebView2 时自动安装依赖。 |
| `HyMT-0.1.3-windows-x64-portable.zip` | 免安装版，完整解压后运行 `hymt-desktop.exe`，系统须已有 WebView2。 |
| `HyMT-Windows-NVIDIA-runtime.zip` | 命令行运行包，保留批量文件翻译和脚本调用方式。 |
| `SHA256SUMS.txt` | 交付文件的 SHA-256 校验值。 |
| `desktop-latest.json`、两个 `.sig` | 应用内更新使用的清单与签名，无需手动打开。 |

要求 Windows x64、受支持的 NVIDIA 显卡及 580.88 或更新驱动，无需另装 Python 或 CUDA Toolkit。NVFP4 用于适配的 RTX 50 架构，其他支持架构使用 Q4_K_M；只有 RTX 5090 经过本机 GPU 验证。

首次运行，在「模型管理」下载或导入模型，启动服务后将 Base URL、模型名和 API Key 填入调用方。默认监听本机 `127.0.0.1`，支持流式响应。模型不包含在附件中，首次下载需要联网，之后可离线使用。

## 桌面功能

- 模型下载、断点续传、GGUF 导入和完整 SHA-256 校验。
- 默认 2K 上下文、Q8 KV、总显存 30% 的估算预算，自动选择模型、显卡和并发并显示实际选择。
- 可自定义上下文与显存预算；设置旁提供说明，展示 KV 缓存节省和性能参考。
- 配置更改后点击「重新部署 · 应用更改」，无需重启桌面程序；小型「恢复默认」按钮保留 API 端口、鉴权与日志设置。
- 托盘、开机自启、静默自启、首次关闭窗口询问，以及安装版和免安装版的签名更新。
- API Key 可关闭或刷新，默认开启鉴权；默认日志仅存内存，也可完全关闭日志或写入轮转文件。
- 退出应用时停止所属推理进程树并释放显存；重新部署预算计入旧服务将释放的显存。
- 紧凑界面，显卡信息并入服务概览，API Key 开关与刷新按钮位于密钥下方。

30% 是包含安全余量的配置估算上限，并非 GPU 驱动层的硬配额。实际占用随负载变化；预算不足时会提示调整参数，不自动超配。

## 模型目录与旧版迁移

- 免安装版优先使用程序旁的 `models`，下载和导入均写入此目录；设置保存在旁边的 `data`，整个文件夹可以搬走。
- 安装版模型使用 `%LOCALAPPDATA%\io.github.divingclone.hymt\models`，不随版本号变化。
- 自动查找旧 `data/models`、附近旧版目录、安装版目录及此前记录的模型位置，校验后复用到当前目录。同盘优先硬链接，跨盘复制，原文件保留。
- 相同权重可复用，不同内容的同名权重分开存储。旧目录未被找到时，手动导入一次对应 GGUF 即可。
- 程序内升级保留数据和模型。0.1.0–0.1.2 本地试用版可安装新版或解压新版后迁移使用。

## 验证与分发说明

通过 105 项 Python 测试、3 项前端配置测试、Svelte/TypeScript 构建检查、Windows 进程树清理测试，以及 RTX 5090 上的 API、流式响应、重新部署和退出清理测试。实际确认免安装版从程序旁的模型目录启动成功。

附件具有 Tauri 更新签名，未使用 Windows Authenticode 代码签名证书。仓库保留完整源码、依赖锁文件、构建说明和第三方许可；模型继续独立托管于 [Hugging Face](https://huggingface.co/divingclone/Hy-MT2-1.8B-NVFP4-Q4_K_M-GGUF)。

[桌面端使用说明](https://github.com/divingclone/Hy-MT2-Windows/blob/desktop-v0.1.3/docs/DESKTOP.md) · [命令行使用说明](https://github.com/divingclone/Hy-MT2-Windows/blob/desktop-v0.1.3/docs/USAGE.md)
