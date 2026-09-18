param(
    [ValidateSet('auto','optimized','baseline','portable')][string]$Variant = 'auto',
    [ValidateSet('auto','fast','official')][string]$Profile = 'auto',
    [string]$Model = '',
    [ValidateRange(0,256)][int]$Parallel = 0,
    [int]$ContextPerSlot = 1024,
    [int]$Batch = 0,
    [int]$MicroBatch = 0,
    [ValidateRange(1,65535)][int]$Port = 18080,
    [string]$Gpu = '',
    [switch]$BackendSampling,
    [switch]$Background,
    [ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$Label = 'server'
)
$ErrorActionPreference = 'Stop'
$launcherArgs = @('--variant', $Variant, '--profile', $Profile, '--context', "$ContextPerSlot", '--port', "$Port", '--label', $Label)
if ($Model) { $launcherArgs += @('--model', $Model) }
if ($Parallel) { $launcherArgs += @('--parallel', "$Parallel") }
if ($Batch) { $launcherArgs += @('--batch', "$Batch") }
if ($MicroBatch) { $launcherArgs += @('--ubatch', "$MicroBatch") }
if ($Gpu) { $launcherArgs += @('--gpu', $Gpu) }
if ($BackendSampling) { $launcherArgs += '--backend-sampling' }
if ($Background) { $launcherArgs += '--background' }
& "$PSScriptRoot/run-python.cmd" "$PSScriptRoot/serve.py" @launcherArgs
exit $LASTEXITCODE
