# Windows 源码构建

发布包可直接运行。只有修改 C++/CUDA 或重建程序才需要本页工具；Python 推理启动器只使用标准库，不需要在发布包的 `runtime/python` 内安装开发依赖。

## 安装构建工具

- Windows x64；Git for Windows；独立安装的 CPython 3.12 x64（带 pip 和 venv）。
- Visual Studio 2022 或 Build Tools 2022，安装“使用 C++ 的桌面开发”、MSVC x64/x86 工具及 Windows SDK。
- 已安装的 NVIDIA CUDA Toolkit 13.x。本次测试使用 CUDA 13.0、MSVC 14.43.34808、Windows SDK 10.0.26100.0；不同工具链组合需要重新验证。
- CMake/Ninja 可使用系统安装，也可按下面的开发环境命令安装。本项目已验证的 Python 工具版本固定在 `requirements-dev.txt`。

编译使用系统安装的工具链，不下载或分发 Visual Studio、Windows SDK、CUDA SDK，也不依赖开发电脑上的 SDK 副本。

在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/prepare_source.py
```

`prepare_source.py` 获取官方 `ggml-org/llama.cpp` 的固定提交 `bdcbaaf6e7520b68c8c60ff724c67409970d70e1`，校验 `patches/hy-mt2.patch` 的 SHA256，应用包含新增文件的补丁，并逐个校验修改文件。默认写入 `src/llama.cpp`；存在该目录时拒绝覆盖。失败会保留临时源码，方便检查；重试使用新的 `--destination`。

Windows 工作区可能使用 CRLF。补丁清单同时记录原始 SHA256 和文本规范为 LF 后的 SHA256，干净源码使用 LF 校验，不会忽略其他内容差异。准备脚本把补丁暂存到这个新仓库的 Git index，不创建提交，也不改动其他 checkout。

## 编译

打开 **Developer PowerShell for VS 2022**，确保 x64 目标环境，再进入仓库。如果快捷方式没有设置 x64，可在该窗口执行：

```powershell
Enter-VsDevShell -VsInstallPath $env:VSINSTALLDIR -DevCmdArguments '-arch=x64 -host_arch=x64' -SkipAutomaticLocation
```

将开发环境的 CMake/Ninja 放到 PATH，然后构建：

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
.\scripts\build-source.ps1 -Jobs 4
```

多版本 CUDA 安装可明确选择目录：

```powershell
.\scripts\build-source.ps1 -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' -Jobs 4
```

如系统阻止本仓库脚本，可仅对本次进程执行 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build-source.ps1 -Jobs 4`；必须从已配置的 Developer PowerShell 调用以继承 MSVC 环境。

默认构建输出 `build/portable/bin`；运行启动器与打包器统一默认使用已安装的 `bin`。验证源码构建时通过 `--binary-dir build/portable/bin` 明确指定程序，验收后将配套 EXE、DLL 和 `build-info.json` 一起安装到 `bin`。构建成功后生成不含本机绝对路径的 `build-info.json`。`-ConfigureOnly` 只配置；`-SourceDir`、`-BuildDir` 可选择独立目录。切换编译器/SDK时应使用新的构建目录，避免复用不兼容的 CMake cache。

固定配置为 CUDA cubin `75/80/86/89/120a`，不嵌入 PTX；CPU 使用 x86-64/SSE2，关闭 native、AVX、AVX2、AVX512 和 BMI2。对应 GTX16/RTX20、RTX30、RTX40、RTX50 的指定计算能力，不代表所有 NVIDIA GPU，也不代表这些代际都经过实机测试。NVFP4 原生 Tensor Core 路径面向 CC12.0；较早显卡由启动器选择 Q4_K_M。模型下载与编译独立，构建不会下载模型。

2026-09-18 已使用本页的 CUDA 13.0、MSVC 14.43.34808 和 Windows SDK 10.0.26100.0 完成五架构构建，并核验实际 DLL 内的 cubin 与无 PTX 配置。当前 `bin` 与受测 EXE/DLL 的 SHA256 一致；RTX 5090 上通过 102/102 项定向 CUDA 算子检查、8/8 项 Q8 attention 检查及 364 次 GPU 前缀/CPU 采样对照，CPU 量化数值测试为 0 失败。其他四种架构仅完成编译覆盖。工具链、日志和产物哈希见[当前构建验收](../benchmarks/current-build-validation.json)。

## 验证和运行

以下命令会实际使用 GPU，建议先关闭其他占用 GPU 的测试任务：

```powershell
$env:PATH = "$PWD\runtime\cuda;$PWD\runtime\msvc;$env:PATH"
.\build\portable\bin\test-backend-ops.exe test -b CUDA0 -o ROPE_RMS_NORM_MUL
.\build\portable\bin\test-backend-ops.exe test -b CUDA0 -o HYMT_PENALTIES
.\build\portable\bin\test-backend-ops.exe test -b CUDA0 -o HYMT_TOP_K_NAN
.\build\portable\bin\test-backend-ops.exe test -b CUDA0 -o GET_ROWS_SCALAR
.\build\portable\bin\test-hymt-prefix.exe
.\translate-batch.cmd examples/input.jsonl results/example.jsonl --binary-dir build/portable/bin --parallel 4
```

首次源码构建若没有发布包的运行 DLL，可将安装目录的 CUDA `bin/x64`（CUDA13）和 MSVC redist x64 目录加入当前 PATH，或安装官方 MSVC x64 运行库。工具链安装的 SDK 目录不是发布包内容。GPU 驱动仍由目标 Windows 系统提供；运行时兼容范围见 [使用与配置](USAGE.md)。

开发时需要量化和校准工具，可另外运行 `cmake --build build/portable --target llama-quantize llama-imatrix hy-logits test-quantize-fns --parallel 4`。模型重排脚本使用 `requirements-dev.txt` 内的 NumPy/PyYAML/tqdm 和固定源码中的 `gguf-py`。不要用通用上游二进制加载这里的融合布局模型。NVFP4 MSE 校准及 Q8/Q4 KV 的完整流程见 [QUANTIZATION.md](QUANTIZATION.md)。

便携包的独立 Python 和运行 DLL 可从已验证的 Releases 环境包取得，或按各自许可从官方发行版准备。打包说明见 [发布流程](RELEASE.md)。`package_windows.py --dry-run` 检查缺失项；源码构建不自动获取或复制可再分发运行库。
