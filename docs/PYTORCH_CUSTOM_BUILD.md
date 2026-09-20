# HyMT 专用 PyTorch CUDA 后端

此构建固定 PyTorch `2.11.0+cu130` 的源码提交
`70d99e998b4955e0049d13a98d77ae1b14db1f45`，只重建 `torch_cuda.dll`。
CPU、c10、c10_cuda、Python 绑定及 vLLM 扩展继续使用原运行时的二进制。
构建产物必须先通过依赖/导出符号审计及实际推理验收，才可用于 Release。
现有普通打包规则不会自动删除这些 CUDA 库。

## 裁剪方式

- cuFFT、cuSPARSE、cuSOLVER 使用 MSVC `/DELAYLOAD`；文本推理不调用这些接口时无需携带对应 DLL。三者在上一版候选 ZIP 中合计约 463 MB 压缩数据，最终净收益还要计入重编后的 CUDA 后端大小变化。
- 三者移除后，其下游 `nvJitLink_130_0.dll` 也可从这个文本推理候选包中略去，另约 36 MB 压缩数据。原生加载引用来自被移除的 FFT/稀疏库；CUDA Python 的专用 `nvjitlink` 模块是未使用的可选接口，`runtime` 绑定中的匹配只是文档链接。FlashInfer 的 CCCL 动态编译路径也不在本项目的 AOT 采样路径内。候选仍须通过实际冷启动验收。
- cuFFT 的 FFTW 兼容包装库 `cufftw64_12.dll` 也一并略去，防止它在导入 PyTorch 时重新强制加载 cuFFT。
- 可选库仅从 `torch_cuda.dll` 同目录加载。缺失时抛出明确的 `c10::Error`，不会伪造算子结果或从宿主机 CUDA 安装目录补齐依赖。它不是完整功能的通用 PyTorch 分发版。
- 默认编译架构保留 SM80、SM86、SM120，去掉不在本项目支持范围的 SM75、SM90、SM100。上游为少量算子单独生成的 SM89、SM120a、SM121a 优化内核仍保留，实际列表由 `cuobjdump` 读取并写入候选记录。
- 关闭未使用的 PyTorch memory-efficient SDPA 和 MAGMA。HyMT 使用 vLLM 的 `TRITON_ATTN`，量化扩展、cuBLAS/cuBLASLt 和实际推理配置保持原样。
- 编译继续使用 Release 优化，不采用 `-Os`、运行时解压或修改精度来换体积。

定制通过 `CMAKE_PROJECT_Torch_INCLUDE` 注入，源码树保持干净。CMake 在上游目标生成后，把 CUDA 后端对 CPU/c10 的链接替换为原 wheel 导入库；保留头文件生成任务，不编译 CPU 后端。只有固定源码与固定原始 wheel 组合可用，升级后需要重新审计 ABI。

## 本机构建

需要 Visual Studio 2022 的 x64 C++ 工具、CUDA Toolkit 13.0、Git、uv，以及项目完整开发运行时。隔离的源码、SDK、构建环境和中间文件均位于 `.local/`，不会安装到系统 Python。

```powershell
./scripts/prepare_pytorch_slim.ps1
./scripts/build_pytorch_slim.ps1 -Jobs 2
```

构建支持中断后增量继续；`-Action Configure` 仅生成构建图。默认双任务限制编译的内存压力。候选 DLL 输出到 `.local/pytorch-source/build-hymt/bin/torch_cuda.dll`。

Windows cuDNN wheel 只有头文件和 DLL，构建脚本从原始 `cudnn64_9.dll` 的公开导出表生成 MSVC 导入库；不会修改 cuDNN 二进制。

## 候选审计

使用已经裁剪过的便携包 `payload` 作为基线，候选目录必须尚不存在：

```powershell
.local/pytorch-build-env/Scripts/python.exe scripts/pytorch/audit_backend.py `
  --dll .local/pytorch-source/build-hymt/bin/torch_cuda.dll `
  --base dist/runtime-dependencies-release/HyMT-0.2.2-windows-x64-portable/payload `
  --output results/pytorch-slim-binary-audit.json `
  --stage dist/pytorch-slim-candidate/payload
```

脚本确认三个库均为延迟导入、其余保留二进制对上述四个大库及 FFTW 包装无强制依赖，并检查所有实际使用的 CUDA 导出符号和原 CPU/c10 导出是否匹配，同时通过 `cuobjdump` 检查 DLL 的实际 GPU 架构。成功后才复制候选运行时、替换一个 DLL、略去五个文件。保留的原生 Python 绑定仍内嵌官方构建信息，因此只修正 `torch.cuda.get_arch_list()` 的一行实现，使公开架构查询返回新 DLL 的真实列表；`torch.__config__` 的原构建信息仍需与定制记录一起解读。

生成的 `torch/hymt-build.json` 记录实际 SHA-256 和基线 CPU/c10/Python DLL 的身份，初始 `inference_validated` 为 `false`。二进制审计通过不代表模型推理或性能验证通过。打包器会拒绝未完成验收或 DLL/架构查询/基线库哈希不匹配的定制运行时。

随后只需针对现有 NVFP4/INT4 配置做必要的冷启动、输出与短热测检查，不重复大规模请求矩阵。验收前不要替换原开发运行时或线上 Release。

验收记录应包含被验证 DLL 的 SHA-256、配置、结果及限制，并随包保留。完成检查后才将候选 `torch/hymt-build.json` 的 `inference_validated` 设为 `true`，同时记录验收报告路径和 SHA-256。使用现有桌面程序生成本地便携包：

```powershell
.local/pytorch-build-env/Scripts/python.exe scripts/pytorch/package_backend.py `
  --payload dist/pytorch-slim-candidate/payload `
  --output dist/pytorch-slim-release
```

该命令执行相同的版本/哈希验收门槛和 ZIP 完整性检查，不重新编译桌面界面，也不修改在线 Release。输出目录应是新的目录；不能将构建源码、SDK、模型或验证缓存放入候选 payload。

MSVC 延迟加载机制参见 [Microsoft 文档](https://learn.microsoft.com/en-us/cpp/build/reference/linker-support-for-delay-loaded-dlls?view=msvc-170)。

## 本次验收范围

RTX 5090，NVFP4 和 INT4，每个运行时每种配置各 32 条输入，冷启动一次、热测两次；2K 上下文、32 并发、INT8 KV。隔离空缓存并禁止宿主构建工具，两种配置共 384 次请求全部完成，192 组结果逐 token 一致；候选未从其他路径加载被删除的 CUDA 库。

| 确定性离线对照 | 原运行时 TPS | 定制运行时 TPS |
| --- | ---: | ---: |
| NVFP4 | 6234.95 | 6235.66 |
| INT4 | 7180.92 | 6988.16 |

输出一致性对照只在验证子进程设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，产品配置保持原样。此前默认多进程 NVFP4 短测的前两轮全部相同，最后一轮 19/32 条输出不同；所有请求均正常结束。默认模式不保证结果可复现，见 [vLLM 官方说明](https://docs.vllm.ai/en/latest/usage/reproducibility/)。受控对照消除了这次观察到的差异，但没有证明此前差异的唯一原因。默认模式 NVFP4 热测中位数为原运行时 5760.66 TPS、定制版 5926.76 TPS。

这些短测只用于发现明显回退，不能证明所有负载性能完全相同，不宣称加速；RTX 30/40 未做实卡验证。原始异常记录保留于 `results/pytorch-slim-acceptance`，受控记录位于 `results/pytorch-slim-controlled`，可分发摘要为 [验收记录](../benchmarks/custom-pytorch-runtime.json)。
