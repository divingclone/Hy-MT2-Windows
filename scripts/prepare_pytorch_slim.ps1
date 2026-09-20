$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot '.local/pytorch-source'
$buildEnv = Join-Path $projectRoot '.local/pytorch-build-env'
$commit = '70d99e998b4955e0049d13a98d77ae1b14db1f45'
if (!(Test-Path -LiteralPath $sourceRoot)) {
    & git clone --filter=blob:none --no-checkout https://github.com/pytorch/pytorch.git $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch source download failed.' }
    & git -C $sourceRoot -c core.longpaths=true checkout $commit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned source checkout failed.' }
}
if ((& git -C $sourceRoot rev-parse HEAD) -ne $commit) {
    throw "Existing checkout must be at $commit; it has not been changed."
}
& git -C $sourceRoot config core.longpaths true
$modules = @('pybind11', 'protobuf', 'gloo', 'pthreadpool', 'FXdiv', 'FP16', 'psimd',
    'cpuinfo', 'onnx', 'sleef', 'ideep', 'fbgemm', 'XNNPACK', 'fmt', 'cudnn_frontend',
    'kineto', 'pocketfft', 'ittapi', 'flatbuffers', 'nlohmann', 'cutlass', 'mimalloc',
    'cpp-httplib', 'NVTX') | ForEach-Object { "third_party/$_" }
& git -C $sourceRoot -c core.longpaths=true submodule update --init --recursive --jobs 4 --depth 1 --filter=blob:none -- @modules
if ($LASTEXITCODE -ne 0) { throw 'PyTorch submodule download failed.' }
if (!(Test-Path -LiteralPath (Join-Path $buildEnv 'Scripts/python.exe'))) {
    & uv venv --python 3.12 $buildEnv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create the isolated build environment.' }
}
# The full build dependency set is pinned. cuDNN is needed for headers only;
# its optional Python-wheel CUDA runtime dependencies come from the base Torch.
& uv pip install --no-deps --python (Join-Path $buildEnv 'Scripts/python.exe') -r (Join-Path $PSScriptRoot 'pytorch/requirements-build.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot install build dependencies.' }
$sdk = Join-Path $projectRoot '.local/pytorch-sdk'
$archive = Join-Path $sdk 'libuv.tar.bz2'
New-Item -ItemType Directory -Force $sdk | Out-Null
if (!(Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest 'https://s3.amazonaws.com/ossci-windows/libuv-1.40.0-h8ffe710_0.tar.bz2' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive).Hash -ne 'E22158FEEA790E9C4E9F14A3DD85D9FDBA8DCDC9B4A7B998E6ADADFE8DEE3865') {
    throw 'libuv archive does not match the pinned SHA-256.'
}
New-Item -ItemType Directory -Force (Join-Path $sdk 'libuv') | Out-Null
& tar -xf $archive -C (Join-Path $sdk 'libuv')
if ($LASTEXITCODE -ne 0) { throw 'Cannot unpack libuv.' }
Write-Host 'Build dependencies ready. Run scripts/build_pytorch_slim.ps1.'
