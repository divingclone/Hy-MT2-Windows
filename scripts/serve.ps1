param(
    [ValidateSet('auto','fast','official')][string]$Profile = 'auto',
    [string]$Model = '',
    [string]$BinaryDir = '',
    [ValidateRange(0,256)][int]$Parallel = 0,
    [int]$ContextPerSlot = 1024,
    [int]$Batch = 0,
    [int]$MicroBatch = 0,
    [ValidateSet('f16','q8_0','q4_0')][string]$CacheTypeK = 'f16',
    [ValidateSet('f16','q8_0','q4_0')][string]$CacheTypeV = 'f16',
    [ValidateRange(1,65535)][int]$Port = 18080,
    [string]$Gpu = '',
    [switch]$BackendSampling,
    [switch]$Background,
    [ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$Label = 'server'
)
$ErrorActionPreference = 'Stop'
$launcherArgs = @('--profile', $Profile, '--context', "$ContextPerSlot", '--port', "$Port", '--label', $Label)
$launcherArgs += @('--cache-type-k', $CacheTypeK, '--cache-type-v', $CacheTypeV)
if ($Model) { $launcherArgs += @('--model', $Model) }
if ($BinaryDir) { $launcherArgs += @('--binary-dir', $BinaryDir) }
if ($Parallel) { $launcherArgs += @('--parallel', "$Parallel") }
if ($Batch) { $launcherArgs += @('--batch', "$Batch") }
if ($MicroBatch) { $launcherArgs += @('--ubatch', "$MicroBatch") }
if ($Gpu) { $launcherArgs += @('--gpu', $Gpu) }
if ($BackendSampling) { $launcherArgs += '--backend-sampling' }
if ($Background) { $launcherArgs += '--background' }
& "$PSScriptRoot/run-python.cmd" "$PSScriptRoot/serve.py" @launcherArgs
exit $LASTEXITCODE
