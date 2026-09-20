# Native Windows vLLM runtime

The application now uses the SystemPanic native Windows build of vLLM 0.29.0
(upstream vLLM Apache-2.0), PyTorch 2.11.0, Transformers, compressed-tensors,
Triton Windows and FlashInfer. Their wheel metadata and license/notice files
are retained with the bundled site-packages. See requirements-vllm-windows.txt
for pinned sources. Only NVIDIA display drivers are a client prerequisite. MSVC redistributable
runtime DLLs and CUDA components supplied by the runtime wheels are bundled;
the MSVC compiler, Windows SDK and full CUDA Toolkit are not redistributed.
Triton supplies TinyCC/PTXAS. FlashInfer sampling is compiled on the build host
for SM80/86/89/120f and distributed as an AOT DLL with provenance.
Microsoft WebView2 uses the system Evergreen runtime when available. Otherwise a
verified Fixed Version runtime is downloaded app-locally under its distribution terms;
see https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution.
Its third-party notices and Microsoft-signed executables are retained.

The local Hunyuan compatibility plugin adapts model registration/weight loading;
the FlashInfer patch only tolerates localized compiler diagnostic encoding.
The NVFP4 checkpoint is derived from Tencent Hy-MT2-1.8B under its model license.
Historical llama.cpp notices remain for archived source only.
