param(
 [ValidateSet('auto','fast','quality','compat')][string]$Profile='auto',
 [string]$Model='', [ValidateRange(1,256)][int]$Parallel=32,
 [int]$ContextPerSlot=2048, [int]$BatchTokens=2048,
 [ValidateSet('int8_per_token_head','bfloat16','fp8_per_token_head')][string]$KVCacheDtype='int8_per_token_head',
 [double]$KVGib=0, [int]$MemoryPercent=75, [int]$Port=18080,
 [string]$Gpu='', [switch]$Background,
 [ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$Label='server'
)
$ErrorActionPreference='Stop'
$launcherArgs=@('--profile',$Profile,'--parallel',"$Parallel",'--context',"$ContextPerSlot",'--batch-tokens',"$BatchTokens",'--kv-cache-dtype',$KVCacheDtype,'--memory-percent',"$MemoryPercent",'--port',"$Port",'--label',$Label)
if ($Model) { $launcherArgs+=@('--model',$Model) }
if ($Gpu) { $launcherArgs+=@('--gpu',$Gpu) }
if ($KVGib) { $launcherArgs+=@('--kv-gib',"$KVGib") }
if ($Background) { $launcherArgs+='--background' }
& "$PSScriptRoot/run-python.cmd" "$PSScriptRoot/serve.py" @launcherArgs
exit $LASTEXITCODE
