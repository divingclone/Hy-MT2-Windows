param(
    [ValidateRange(1,64)][int]$Jobs = 4,
    [string]$SourceDir = '',
    [string]$BuildDir = '',
    [string]$CudaPath = $env:CUDA_PATH,
    [switch]$ConfigureOnly
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (!$SourceDir) { $SourceDir = Join-Path $root 'src/llama.cpp' }
if (!$BuildDir) { $BuildDir = Join-Path $root 'build/portable' }
$source = (Resolve-Path -LiteralPath $SourceDir).Path
$build = [IO.Path]::GetFullPath($BuildDir)
$pin = 'bdcbaaf6e7520b68c8c60ff724c67409970d70e1'

function Find-Tool([string]$Name) {
    $tool = Get-Command $Name -CommandType Application -ErrorAction Stop | Select-Object -First 1
    return $tool.Source
}
function Invoke-Tool([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

$cmake = Find-Tool 'cmake.exe'
$ninja = Find-Tool 'ninja.exe'
$git = Find-Tool 'git.exe'
$cl = Find-Tool 'cl.exe'
$rc = Find-Tool 'rc.exe'
$mt = Find-Tool 'mt.exe'
if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64' -or !$env:INCLUDE -or !$env:LIB) {
    throw 'Run in VS 2022 Developer PowerShell with x64 target (see docs/BUILD.md).'
}
if (!$CudaPath) { $CudaPath = Split-Path -Parent (Split-Path -Parent (Find-Tool 'nvcc.exe')) }
$cuda = (Resolve-Path -LiteralPath $CudaPath).Path
$nvcc = Join-Path $cuda 'bin/nvcc.exe'
if (!(Test-Path -LiteralPath $nvcc -PathType Leaf)) { throw 'CUDA Toolkit nvcc.exe is missing.' }
$cudaVersion = (& $nvcc --version) -join "`n"
if ($LASTEXITCODE -ne 0 -or $cudaVersion -notmatch 'release (13\.[0-9]+)') {
    throw 'This build recipe requires an installed CUDA 13.x Toolkit.'
}
$cudaRelease = $Matches[1]
$commit = (& $git -C $source rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $commit -ne $pin) { throw 'Source HEAD must match the pinned commit; run prepare_source.py.' }
if (!(Test-Path -LiteralPath (Join-Path $source 'tools/hy-batch/CMakeLists.txt'))) {
    throw 'The Hy-MT2 patch is missing; run prepare_source.py into a new source directory.'
}
if ($build -eq $source -or $source.StartsWith($build.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'BuildDir must be separate from, and not contain, SourceDir.'
}
$cacheFile = Join-Path $build 'CMakeCache.txt'
if (Test-Path -LiteralPath $cacheFile) {
    $cache = Get-Content -LiteralPath $cacheFile -Raw
    foreach ($entry in @(@('CMAKE_C_COMPILER', $cl), @('CMAKE_CXX_COMPILER', $cl),
                         @('CMAKE_CUDA_COMPILER', $nvcc), @('CMAKE_CUDA_HOST_COMPILER', $cl),
                         @('CMAKE_HOME_DIRECTORY', $source))) {
        $pattern = '(?m)^' + $entry[0] + ':[^=]+=(.+)\r?$'
        if ($cache -notmatch $pattern -or $Matches[1].Trim().Replace('\','/') -ne $entry[1].Replace('\','/')) {
            throw 'Existing CMake cache belongs to another compiler/source; choose a new -BuildDir.'
        }
    }
}

# Use the developer environment, but never inherit machine-specific ISA flags.
$cleared = @('CL', '_CL_', 'CFLAGS', 'CXXFLAGS', 'CUDAFLAGS', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')
$saved = @{}
foreach ($name in $cleared) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
try {
    $options = @(
        '-S', $source, '-B', $build, '-G', 'Ninja',
        "-DCMAKE_C_COMPILER:FILEPATH=$($cl.Replace('\','/'))",
        "-DCMAKE_CXX_COMPILER:FILEPATH=$($cl.Replace('\','/'))",
        "-DCMAKE_CUDA_COMPILER:FILEPATH=$($nvcc.Replace('\','/'))",
        "-DCMAKE_CUDA_HOST_COMPILER:FILEPATH=$($cl.Replace('\','/'))",
        "-DCMAKE_RC_COMPILER:FILEPATH=$($rc.Replace('\','/'))", "-DCMAKE_MT:FILEPATH=$($mt.Replace('\','/'))",
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=$($ninja.Replace('\','/'))", "-DCUDAToolkit_ROOT:PATH=$($cuda.Replace('\','/'))",
        "-DGIT_EXECUTABLE:FILEPATH=$($git.Replace('\','/'))", '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CUDA_ARCHITECTURES:STRING=75-real;80-real;86-real;89-real;120a-real',
        '-DCMAKE_CUDA_FLAGS:STRING=-D_WINDOWS -Xcompiler=/EHsc -Xcompiler=/utf-8 --use-local-env',
        '-DGGML_CUDA=ON', '-DGGML_CUDA_GRAPHS=ON', '-DGGML_CUDA_COMPRESSION_MODE=size',
        '-DGGML_CUDA_CUB_3DOT2=OFF', '-DGGML_NATIVE=OFF', '-DGGML_CPU_ALL_VARIANTS=OFF',
        '-DGGML_SSE42=OFF', '-DGGML_AVX=OFF', '-DGGML_AVX2=OFF', '-DGGML_BMI2=OFF',
        '-DGGML_AVX_VNNI=OFF', '-DGGML_AVX512=OFF', '-DGGML_AVX512_VBMI=OFF',
        '-DGGML_AVX512_VNNI=OFF', '-DGGML_AVX512_BF16=OFF', '-DLLAMA_OPENSSL=OFF',
        '-DLLAMA_BUILD_TESTS=ON', '-DLLAMA_BUILD_EXAMPLES=ON', '-DLLAMA_BUILD_SERVER=ON'
    )
    Invoke-Tool $cmake $options
    if ($ConfigureOnly) { return }
    Invoke-Tool $cmake @('--build', $build, '--parallel', "$Jobs", '--target',
                        'hy-batch', 'llama-server', 'llama-bench', 'hy-logits', 'test-hymt-prefix', 'test-backend-ops')
    $dirty = [bool]((& $git -C $source status --porcelain) -join "`n")
    if ($LASTEXITCODE -ne 0) { throw 'Cannot record source status.' }
    $manifest = Get-Content -LiteralPath (Join-Path $root 'patches/hy-mt2.manifest.json') -Raw | ConvertFrom-Json
    $info = [ordered]@{
        schema_version = 1; platform = 'windows-x86_64'; cuda_toolkit = $cudaRelease
        architectures = @('75','80','86','89','120a'); ptx_architectures = @()
        compute_capabilities = @('7.5','8.0','8.6','8.9','12.0'); minimum_driver = '580.88'
        minimum_driver_policy = 'Conservative Windows package floor; CUDA 13.x requires R580 or newer.'
        nvfp4_compute_capabilities = @('12.0'); cpu_baseline = 'x86-64/SSE2'
        ggml_native = $false; cpu_avx = $false; cpu_avx2 = $false; cpu_avx512 = $false
        source_commit = $commit; source_dirty = $dirty; distributed_patch_sha256 = $manifest.patch_sha256
        build_type = 'Release'; built_at_utc = [DateTime]::UtcNow.ToString('o'); validated_devices = @()
        driver_reference = 'https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html'
    }
    $output = Join-Path $build 'bin/build-info.json'
    [IO.File]::WriteAllText($output, ($info | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))
    Write-Output "Built: $build\bin"
    Write-Output 'Build metadata records compiled targets, not tests on every GPU generation.'
}
finally {
    foreach ($name in $cleared) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
