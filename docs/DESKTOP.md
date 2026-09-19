# HyMT 桌面端

HyMT 桌面端只管理翻译推理后端，不提供文本翻译界面。其他应用通过 OpenAI 兼容 API 调用模型。当前目标平台为 Windows x64 + NVIDIA；CUDA 架构及驱动要求与原运行包一致。

## 使用

1. 运行 `HyMT_版本_x64-setup.exe` 安装，或完整解压 `HyMT-版本-windows-x64-portable.zip` 后双击 `hymt-desktop.exe`。
2. 在「模型管理」下载模型，也可导入原运行包的对应 `fused.gguf` 文件。下载与导入都会验证清单中的完整 SHA-256。模型不包含在默认安装包里。
3. 在「服务概览」选择模型、GPU、上下文和并发，点击「启动服务」。默认使用 2K（2048 tokens）上下文、Q8 KV 和总显存 30% 的预算上限，自动推荐并发；自动选项同时显示当前推荐的模型、显卡和参数。上下文可从常用值选择，或选择「自定义」输入 256–32768 之间的整数。设置旁的问号支持鼠标悬停和键盘聚焦查看说明。
4. 将界面提供的 API Base URL、模型名 `hy-mt2` 和 API Key 填入调用方应用。协议支持 `/v1/models`、`/v1/chat/completions` 和流式响应。Base URL 包含 `/v1`，请遵循调用方的填写要求，避免重复添加。

服务仅监听 `127.0.0.1`，没有网页翻译界面。API Key 首次生成后保存在用户数据目录，重启服务保持一致。默认请求需要 `Authorization: Bearer <API Key>`。可关闭「API Key 鉴权」并重新部署，使本机调用无需密钥；也可点击 API Key 下方的「刷新密钥」轮换密钥并重新部署，旧密钥失效后需同步修改调用方。密钥不显示在日志里，可点击复制。

默认端口 18080。如果被占用，会尝试接下来的 19 个端口，并在界面显示实际地址；客户端须使用显示的实际地址。不会接管或停止已有的其他推理进程。

显存预算上限可在右侧调节为总显存的 10%–100%，默认 30%；以该上限和重部署可用显存两者的较小值计算。自动和手动并发都须通过预算检查，不足时提示调高比例或降低配置，不自动超出设定预算。配置标题旁的「恢复默认」恢复推理参数，保留 API 端口、鉴权和日志设置，运行中的服务须重新部署后生效。

显存预算包含模型预留、按 256 token 对齐的 KV、计算工作区与安全余量。它是估计，不是实际占用保证。编辑配置时预算随之预览，每 10 秒刷新当前空闲显存；「重部署可用」包含本应用管理的服务将释放的显存，不含其他服务，也不重复计入安全余量。驱动支持时读取进程的实际显存，WDDM 返回 N/A 时使用本服务启动前后显卡空闲量之差并标记为估计。

修改配置后出现「重新部署 · 应用更改」按钮，无需退出桌面程序。先验证新配置和模型，检查失败时原服务继续运行；检查通过后结束旧进程树，再用实际释放后的显存重新检查并启动新服务。部署期间 API 短暂中断；显存变化或启动失败时显示错误，可修改后重试。保存仅保存下次启动的配置，不会暗中中断当前请求。

Q8/Q4 显存节省相对相同上下文、并发的 F16 计算，含块缩放开销：KV 部分分别减少 46.875% / 71.875%，界面同时显示节约的 GiB 与总预算比例。速度提示引用项目实测：Q8 的 13.6% 来自 RTX 5090 当前 NVFP4、256 并发原生批处理；Q4 的 14.6% 来自旧实验 NVFP4、128 并发（`benchmarks/nvfp4-kv-evaluation.json`）。测试条件不同，不能据此比较两种缓存速度，不能解释为当前 HTTP API 性能预测。其他显卡或 Q4_K_M 没有相应实测，量化 KV 的实际质量与速度须按业务验证。

## 托盘、开机自启和退出

- 最小化默认进入系统托盘，推理继续运行；点击托盘图标恢复窗口。
- 第一次关闭窗口时，通过 Tauri 官方 dialog 询问「退出程序」「最小化到托盘」或「取消」。前两项会记住选择，取消不记住。可随时在应用设置修改「关闭窗口时保留在托盘」。退出程序终止推理服务及其子进程；托盘模式下 API 继续运行，直到使用托盘菜单或页面底部「退出应用并停止服务」。
- 「开机自启」使用 Tauri 官方 autostart 插件注册；「静默自启」仅在系统登录启动时隐藏窗口，手动打开仍显示界面。
- 「打开应用时启动推理」独立控制是否自动加载模型。静默启动失败时显示窗口，不静默循环重试占用 GPU。
- 单实例插件阻止同一应用重复打开和重复加载模型。
- Windows Job Object 将工作进程及 llama-server 子进程绑定到桌面进程。工作进程先等待 stdin 握手，只有绑定 Job 成功后才允许启动。正常退出、强制结束桌面进程或崩溃都会由 Windows 结束整个任务树。

## 数据与错误处理

安装版的数据位于 `%LOCALAPPDATA%\io.github.divingclone.hymt`，模型使用其中固定的 `models` 目录，不随版本号变化。免安装版优先将模型下载或导入到可执行文件旁的 `models` 文件夹，设置、密钥及 WebView2 数据仍位于旁边的 `data`；搬走整个文件夹即可继续使用。模型页显示实际路径。权重按 SHA-256 子目录存放，相同内容可复用，不同内容的同名模型不会相互覆盖。不要放在只读目录。Tauri log 插件关闭默认文件目标，避免在内存模式下额外写入应用日志。

首次使用会在当前程序、旧 `data/models`、附近旧版 HyMT 目录、安装版目录及此前记录的模型目录中查找已有模型。完整 SHA-256 校验通过后优先创建硬链接，跨磁盘则复制到当前版本的模型目录，原文件保留；不会全盘扫描。不同版本在用户数据目录的 `model-locations.json` 中记录模型位置，仅保存路径，免安装版本地运行不依赖此索引。旧版放在未记录的其他位置时，在模型页导入对应 GGUF 一次即可。删除模型仅影响当前目录中的文件，不删除旧目录原文件，也不会自动重新导入已删除的模型。

模型独立于程序资源，程序内升级时保留。下载支持 HTTP Range 续传、网络失败退避重试、完整校验和原子替换；暂停保留 `.part`，取消导入不会修改已有模型。模型删除和文件替换要求先停止服务。

「运行诊断」提供日志与常见错误处理建议。默认开启「不写入日志文件」，仅在有界内存缓冲保留最近 512 行，每行最多 8,000 字符；退出即清空。关闭该选项后额外写入轮转文件，最多 3 份 × 2 MiB。「完全关闭日志」默认关闭，开启后不收集推理日志（仍保留状态和错误提示）。更换日志模式需要重新部署，不会自动删除此前已有的日志文件。启动会检查驱动、GPU 架构、模型大小和 SHA-256、KV 支持和显存预算；超时或启动失败后结束服务，避免残留进程。

推理 API 不会自动修改调用方提示词。Hy-MT2 是翻译模型，调用方应按模型要求构建翻译提示。客户端应自行处理长文本分段和上下文长度。

## 开发与构建

开发环境需要 Node.js、Rust MSVC 工具链、Visual Studio C++ 构建工具与 WebView2。随包推理库、Python 和 CUDA 运行库需要先按原项目构建说明准备。

```powershell
cd desktop
npm ci
npm run tauri dev
```

开发版从项目根目录读取推理资源，模型仍放在独立用户数据目录，可在界面导入。桌面实现复用 `gpu_config.py` 和 `setup_model.py`，不需要 pip。

```powershell
# 在项目根目录，以 Python 3.12 运行
python scripts/build_desktop.py
# 仅准备资源（可随后自行运行 Tauri CLI）
python scripts/build_desktop.py --stage-only
```

构建脚本复用现有运行包白名单与 PE 依赖检查，把 700 余个运行时文件置于 `desktop/src-tauri/resources/payload`，验证 Python 可迁移性，生成依赖许可文件，再构建 NSIS 安装包、免安装 ZIP 和 SHA256SUMS。输出默认位于 `dist/desktop`；重复打包须选择新的 `--output`，避免覆盖已有发布产物。

安装程序使用官方 NSIS 打包器，按当前用户安装。缺少 WebView2 时会下载微软 bootstrapper。免安装版要求系统已有 WebView2；缺少时可改用安装版。Windows 自身代码签名证书与下述更新签名不同，本仓库未配置 Authenticode 证书。

## 程序内更新与发布

安装版使用 Tauri 官方 updater 执行签名验证和安装。免安装版使用同一个插件下载并验证签名，再由小型 Python 更新助手更换 EXE 和 payload。官方插件的 Windows 安装流程针对安装程序，因此免安装文件替换是必要的定制逻辑：先完整展开新包，退出应用后替换，保留 `data` 和程序旁的 `models`，替换失败尝试回滚；结果位于 `data/updates/result.json`，上一版保留在 `data/updates/previous`。

当前公钥已写入配置。首次本机构建生成的私钥在 `.local/desktop-signing/hymt.key`，没有加入 Git；维护者必须保存该私钥以签署后续更新，不能重新生成一个不同密钥来替代已发布版本所信任的密钥。

```powershell
$env:TAURI_SIGNING_PRIVATE_KEY = (Resolve-Path .local/desktop-signing/hymt.key).Path
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = ''
$env:HYMT_UPDATER_PUBLIC_KEY = (Get-Content .local/desktop-signing/hymt.key.pub -Raw).Trim()
python scripts/build_desktop.py --signed --output dist/desktop-release
```

新维护环境可使用 `npm run tauri signer generate -- --ci -w <私钥路径>` 创建自己的密钥对，并在首次分发前设置公钥。禁止提交私钥。

发布时将安装 EXE、免安装 ZIP、它们的 `.sig`、`desktop-latest.json` 和 SHA256SUMS 上传到同一个 GitHub Release。默认资源地址指向 `desktop-v<版本>` 标签，检查地址为 GitHub 最新 Release 的 `desktop-latest.json`。可通过 `HYMT_RELEASE_BASE_URL`、`HYMT_UPDATE_URL` 改为其他 HTTPS 托管地址。升级三个版本号：`desktop/package.json`、`desktop/src-tauri/Cargo.toml`、`desktop/src-tauri/tauri.conf.json`，并更新锁文件。

生成文件并不等于发布：在更新清单上传前，「检查更新」会显示网络/404 错误，无法完成线上升级。构建脚本不自动创建 Release，不修改远程仓库。

## 验证

```powershell
python -m unittest discover -s tests -v
cd desktop
npm run build
node --test tests/config.test.mjs
cargo test --manifest-path src-tauri/Cargo.toml
# 需要本机 NVIDIA GPU、当前 Q4_K_M 权重及运行库，测试实际 API / SSE / 退出清理
cargo test --manifest-path src-tauri/Cargo.toml real_gpu_api_and_stop -- --ignored --nocapture
```

官方组件：Tauri tray / NSIS；插件 single-instance、autostart、store、updater、dialog、clipboard-manager、opener、log。为保障崩溃清理，后台进程使用原生 Windows Job Object；不向前端开放任意 shell 执行权限。
