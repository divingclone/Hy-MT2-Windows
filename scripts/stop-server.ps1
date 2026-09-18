param([ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$Label = 'server')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "results/$Label.pid"
if (!(Test-Path -LiteralPath $pidFile)) { Write-Output 'No managed server PID file.'; exit 0 }
$serverProcessId = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$process = Get-Process -Id $serverProcessId -ErrorAction SilentlyContinue
if (!$process) { Remove-Item -LiteralPath $pidFile; Write-Output 'Managed server already stopped.'; exit 0 }
$allowed = @('bin/llama-server.exe', 'build/portable/bin/llama-server.exe', 'build/optimized/bin/llama-server.exe', 'build/baseline/bin/llama-server.exe') | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $root $_)) }
if ($process.Path -notin $allowed) { throw 'PID belongs to another executable; refusing to stop it.' }
Stop-Process -Id $serverProcessId
Remove-Item -LiteralPath $pidFile
Write-Output "Stopped managed server PID $serverProcessId."
