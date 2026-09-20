# Dot-source before running the native Windows vLLM experiments. No global
# Windows settings or system Python packages are modified.
param([string]$EnvironmentPath = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (!$EnvironmentPath) { $EnvironmentPath = Join-Path $projectRoot '.local/vllm-win' }
$environmentScripts = Join-Path $EnvironmentPath 'Scripts'
if (!(Test-Path -LiteralPath (Join-Path $environmentScripts 'python.exe'))) {
    throw "Missing vLLM environment: $EnvironmentPath"
}
$vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (!(Test-Path -LiteralPath $vswherePath)) {
    throw 'Install Visual Studio C++ Build Tools for FlashInfer/Triton JIT compilation.'
}
$visualStudioRoot = & $vswherePath -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$visualStudioRoot) { throw 'Visual Studio x64 C++ compiler was not found.' }
$compilerSetup = Join-Path $visualStudioRoot 'VC/Auxiliary/Build/vcvars64.bat'
$compilerEnvironment = & $env:ComSpec /d /s /c ('"' + $compilerSetup + '" >nul && set')
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the MSVC environment.' }
foreach ($line in $compilerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -LiteralPath ('Env:' + $matches[1]) -Value $matches[2]
    }
}
$env:PATH = (Resolve-Path -LiteralPath $environmentScripts).Path + ';' + $env:PATH
$env:PYTHONUTF8 = '1'
$env:VLLM_HOST_IP = '127.0.0.1'
$env:MAX_JOBS = '8'
$env:HYMT_VLLM_NATIVE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
# FlashInfer captures compiler output as UTF-8; localized MSVC output can use
# the Windows ANSI code page and fail decoding even after a successful build.
$env:VSLANG = '1033'
