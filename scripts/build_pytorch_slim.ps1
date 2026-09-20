param(
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [ValidateRange(1, 8)][int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot '.local/pytorch-source'
$buildRoot = Join-Path $sourceRoot 'build-hymt'
$buildEnv = Join-Path $projectRoot '.local/pytorch-build-env'
$baseTorch = Join-Path $projectRoot 'runtime/vllm/Lib/site-packages/torch'
$expectedCommit = '70d99e998b4955e0049d13a98d77ae1b14db1f45'
$actualCommit = & git -C $sourceRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "The PyTorch checkout must be exactly $expectedCommit"
}
if (& git -c core.longpaths=true -C $sourceRoot status --porcelain --untracked-files=no --ignore-submodules=none) {
    throw 'The pinned PyTorch checkout must be unmodified; customization is injected externally.'
}
$version = & (Join-Path $projectRoot 'runtime/vllm/python.exe') -B -c 'import torch; print(torch.__version__); print(torch.version.git_version)'
if ($LASTEXITCODE -ne 0 -or $version[0] -ne '2.11.0+cu130' -or $version[1] -ne $expectedCommit) {
    throw 'The base runtime must be the matching official PyTorch 2.11.0+cu130 build.'
}
. (Join-Path $PSScriptRoot 'vllm_windows_env.ps1') -EnvironmentPath $buildEnv
$env:MAX_JOBS = "$Jobs"
$env:HYMT_BASE_TORCH = $baseTorch
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:libuv_ROOT = Join-Path $projectRoot '.local/pytorch-sdk/libuv/Library'
$cudnn = Join-Path $buildEnv 'Lib/site-packages/nvidia/cudnn'
$python = Join-Path $buildEnv 'Scripts/python.exe'
$cmake = Join-Path $buildEnv 'Scripts/cmake.exe'
$ninja = Join-Path $buildEnv 'Scripts/ninja.exe'
$env:TORCH_CUDA_ARCH_LIST = '8.0;8.6;12.0'
$env:TORCH_NVCC_FLAGS = '-Xfatbin -compress-all'
$env:PYTHONPATH = $sourceRoot
$env:VSLANG = '1033'

# NVIDIA's Windows cuDNN wheel provides headers/DLLs but no MSVC import lib.
# Generate it from the original DLL's public exports (no binary modification).
$cudnnLib = Join-Path $projectRoot '.local/pytorch-sdk/cudnn'
New-Item -ItemType Directory -Force $cudnnLib | Out-Null
$exportLines = & dumpbin /nologo /exports (Join-Path $baseTorch 'lib/cudnn64_9.dll')
if ($LASTEXITCODE -ne 0) { throw 'Cannot read cuDNN exports.' }
$symbols = @($exportLines | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\w+)\s*$') { $matches[1] }
})
if ($symbols.Count -lt 100 -or $symbols -notcontains 'cudnnGetVersion') {
    throw 'Unexpected cuDNN export table.'
}
@('LIBRARY cudnn64_9.dll', 'EXPORTS') + $symbols | Set-Content -Encoding ascii (Join-Path $cudnnLib 'cudnn.def')
& lib /nologo /machine:x64 "/def:$cudnnLib/cudnn.def" "/out:$cudnnLib/cudnn.lib"
if ($LASTEXITCODE -ne 0) { throw 'Cannot create the cuDNN import library.' }

# Keep the original optimization level; only remove unused GPU architectures
# and the unused built-in SDPA implementation. HyMT uses vLLM TRITON_ATTN.
$configureArgs = @(
    '-S', $sourceRoot, '-B', $buildRoot, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
    "-DCMAKE_MAKE_PROGRAM=$ninja", "-DPython_EXECUTABLE=$python",
    "-DCMAKE_PROJECT_Torch_INCLUDE=$PSScriptRoot/pytorch/slim.cmake",
    "-DCMAKE_PREFIX_PATH=$buildEnv/Library", "-DINTEL_MKL_DIR=$buildEnv/Library",
    "-DINTEL_OMP_DIR=$buildEnv/Library", '-DBLAS=MKL', '-DUSE_MKL=ON',
    "-DCUDNN_INCLUDE_DIR=$cudnn/include", "-DCUDNN_LIBRARY=$cudnnLib",
    '-DTORCH_BUILD_VERSION=2.11.0', '-DBUILD_PYTHON=OFF', '-DBUILD_TEST=OFF',
    '-DBUILD_BINARY=OFF', '-DUSE_CUDA=ON', '-DUSE_CUDNN=ON',
    '-DUSE_DISTRIBUTED=ON', '-DUSE_GLOO=ON', '-DUSE_MPI=OFF', '-DUSE_NCCL=OFF',
    '-DUSE_KINETO=ON', '-DUSE_MKLDNN=ON', '-DUSE_OPENMP=ON',
    '-DUSE_NNPACK=OFF', '-DUSE_MAGMA=OFF', '-DUSE_FLASH_ATTENTION=OFF',
    '-DUSE_MEM_EFF_ATTENTION=OFF', '-DUSE_CUSPARSELT=OFF', '-DUSE_CUFILE=OFF',
    '-DBUILD_LAZY_CUDA_LINALG=OFF', '-DONNX_ML=ON', '-DONNX_NAMESPACE=onnx_torch',
    '-DMSVC_Z7_OVERRIDE=OFF', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'
)
New-Item -ItemType Directory -Force $buildRoot | Out-Null
& $cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA configuration failed.' }
if ($Action -eq 'Configure') { return }

# Reject an accidental full CPU rebuild before invoking the compiler.
$dryRun = & $ninja -C $buildRoot -n torch_cuda
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the CUDA build graph.' }
if ($dryRun -match 'Building .* object .*[/\\]torch_cpu\.dir[/\\]') {
    throw 'The build graph unexpectedly rebuilds torch_cpu.'
}
& $cmake --build $buildRoot --target torch_cuda --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA build failed; rerun to resume incrementally.' }
$baseLibraries = @{}
foreach ($name in @('torch_cpu.dll', 'c10.dll', 'c10_cuda.dll', 'torch_python.dll')) {
    $baseLibraries[$name] = (Get-FileHash -LiteralPath (Join-Path $baseTorch "lib/$name")).Hash.ToLowerInvariant()
}
$customization = @{}
foreach ($name in @('slim.cmake', 'delay_load.cpp')) {
    $customization[$name] = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot "pytorch/$name")).Hash.ToLowerInvariant()
}
@{
    source_commit = $expectedCommit
    custom_dll_sha256 = (Get-FileHash -LiteralPath "$buildRoot/bin/torch_cuda.dll").Hash.ToLowerInvariant()
    base_libraries = $baseLibraries
    customization = $customization
    configuration = @($configureArgs | ForEach-Object { $_.Replace($projectRoot, '<project>') })
    built_at = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 "$buildRoot/bin/hymt-build.json"
Write-Host "Built $buildRoot/bin/torch_cuda.dll. Audit and stage into an isolated runtime before packaging."
