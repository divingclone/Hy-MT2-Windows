# HyMT 0.2.2 桌面端

Tauri 2 + Svelte 管理 Windows 原生 vLLM。本轮提供完整便携 ZIP：解压后启动 `hymt-desktop.exe`，模型从 Hugging Face 按文件单独下载并校验复用；源码开发可设置 `HYMT_RUNTIME_ROOT` 指向已准备好的项目根目录。

默认 NVFP4 权重、速度优先 CUTLASS、INT8 KV、2K 上下文、总显存 75% 估算预算，自动并发为 32，显式最大 256。保真优先使用同一模型的 Marlin 16 位激活。UI 显示共享 KV token 池容量，避免把并发误解为完整上下文槽位数。

模型页从两个 Hugging Face 仓库分别下载 NVFP4 或普通 INT4 的 safetensors、配置和 tokenizer，逐文件校验并支持续传，全部完成后原子发布目录。也可导入对应离线 ZIP，验证后原子解压。旧 GGUF 模型不能复用；端口、鉴权、日志和上下文设置保留，旧缓存类型映射到新类型。程序升级不携带模型，已有模型校验后继续复用。

保留托盘、静默启动、API Key 刷新、日志模式、签名便携更新和无效配置预检。GPU 服务由桌面 Windows Job Object 管理，包含 vLLM API/engine/worker 全部后代；退出或强制关闭 GUI 会清理进程树。服务健康判断兼容 vLLM 的空体 HTTP 200。

宿主机要求见 [README](../README.md)。运行包包含 Python、MSVC/CUDA 运行库、TinyCC/PTXAS、预编译采样内核；优先使用系统 WebView2，缺失时自动下载应用本地版本，用户无需手动安装开发工具或界面运行时。RTX 30/40 自动选普通 INT4，尚待对应实卡验证。旧线上 0.1.3 Release 仍为 llama.cpp，本次没有上传新 Release。
