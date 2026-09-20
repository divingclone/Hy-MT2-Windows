# 构建 Windows 原生 vLLM 运行包

依赖固定于 `requirements-vllm-windows.txt`。使用社区 [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows/releases/tag/v0.29.0) wheel；官方上游与此社区 Windows 构建须区分。NVFP4 原生模型插件只支持此验证版本。

仅构建机安装 MSVC x64 C++ Build Tools、Windows SDK、CUDA Toolkit；项目验证 CUDA Toolkit 13.0 + 驱动 596.36。驱动支持下限是本项目的保守实测政策，不将 Linux 版本号误作 Windows 最低要求。CUDA 官方兼容信息见 [NVIDIA 发行说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)。

```powershell
uv venv --python 3.12 .local/vllm-build
uv pip install --python .local/vllm-build/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match -r requirements-vllm-windows.txt
scripts/run-python.cmd scripts/prepare_vllm_runtime.py --from-environment .local/vllm-build --msvc-redist "C:/path/to/VC/Redist/MSVC/14.xx/x64/Microsoft.VC143.CRT"
# 加载构建机 MSVC 环境，然后一次性预编译采样内核
. scripts/vllm_windows_env.ps1
$env:PATH = "$PWD/runtime/vllm/Scripts;" + $env:PATH
$env:CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
$env:CUDA_HOME = $env:CUDA_PATH
runtime/vllm/python.exe scripts/build_flashinfer_aot.py --runtime runtime/vllm --cache .local/flashinfer-aot
# 仅离线完整包需要提前准备固定版浏览器
scripts/run-python.cmd scripts/prepare_webview_runtime.py
```

`prepare_vllm_runtime.py` 将 CPython 基础安装及依赖复制到 `runtime/vllm/`，直接安装 Hunyuan 插件模块和 entry point，移除 editable 导入与源环境路径；使用 `python -m vllm.entrypoints.cli.main`，不依赖 venv 中带绝对路径的 vllm.exe 启动器。FlashInfer 的编译器日志只修改解码容错，不修改计算。

准备模型目录和文件摘要后，用 `scripts/publish_vllm_models.py` 把 safetensors、配置和 tokenizer 直接上传到两个独立 Hugging Face 仓库，核对远程文件后固定提交。`scripts/build_model_bundle.py` 仅为可选离线导入构建 ZIP。完成后再构建程序包。量化入口为 `scripts/quantize_vllm_nvfp4.py`，BF16 源模型仅用于量化和评测，不随默认部署一起加载。

```powershell
scripts/run-python.cmd scripts/package_windows.py --output dist/my-vllm-runtime
python scripts/build_desktop.py --output dist/desktop-vllm
```

便携包包含 vLLM、PyTorch、Python helper、模型下载清单、应用本地 MSVC/CUDA 运行库、TinyCC/PTXAS、预编译采样内核；WebView2 优先使用系统版本，缺失时自动下载。模型通过 Hugging Face 单独下载；默认不打包权重。如需额外构建离线包，可显式添加 `--include-models --include-webview2`。不包含完整 MSVC/CUDA SDK、API Key、缓存、旧 llama.cpp 或开发环境。ZIP 使用 Zip64。桌面构建使用 `--no-bundle` 生成 EXE，再制作完整便携 ZIP；旧的小体积 NSIS 打包流程已停用。

## 检查

```powershell
python -m unittest discover -s tests -v
cd desktop
npm ci
npm run build
node --test tests/config.test.mjs
cd ..
cargo test --locked --manifest-path desktop/src-tauri/Cargo.toml
# 需要实际 GPU，先停其他推理服务
cargo test --locked --manifest-path desktop/src-tauri/Cargo.toml real_gpu_api_and_stop -- --ignored --nocapture
```

迁移后的旧源码重现工具位于 `archive/llama-cpp`，不再作为新后端的构建前置条件。

便携运行时搬迁验收（先停其他推理服务）：

```powershell
python scripts/validate_portable.py --directory dist/desktop-vllm/HyMT-0.2.2-windows-x64-portable --report results/portable-vllm.json
```

免安装冷缓存检查：

```powershell
python scripts/validate_portable_toolchain.py --output results/new-cold-fast
python scripts/validate_portable_toolchain.py --profile compat --output results/new-cold-int4
```

两个 Python 前缀（`runtime/vllm` 与 helper 的 `runtime/python`）都需要应用本地 CRT DLL。桌面构建把 vllm 前缀的 `vcruntime140.dll`、`vcruntime140_1.dll`、`msvcp140.dll` 复制到 EXE 旁边。WebView2 下载由固定 URL、CAB SHA256 和 Microsoft Authenticode 签名校验；固定版文件仅解压，不调用安装器。发布前对照 `licenses/WEBVIEW2-FIXED-LICENSE.txt` 保留分发条款，并更新固定版安全版本。
