param([ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$Label = 'server')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "results/$Label.pid"
if (!(Test-Path -LiteralPath $pidFile)) { Write-Output 'No managed server PID file.'; exit 0 }
$serverProcessId = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
if ($serverProcessId -le 0) { throw 'Invalid managed server PID; refusing to stop it.' }
$process = Get-Process -Id $serverProcessId -ErrorAction SilentlyContinue
if (!$process) { Remove-Item -LiteralPath $pidFile; Write-Output 'Managed server already stopped.'; exit 0 }
$configFile = Join-Path $root "results/$Label.config.json"
if (!(Test-Path -LiteralPath $configFile -PathType Leaf)) { throw 'Missing managed server config; refusing to stop it.' }
$config = Get-Content -LiteralPath $configFile -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.mode -ne 'server' -or ($null -ne $config.label -and $config.label -cne $Label) -or
    $config.executable -isnot [string] -or $config.executable -notmatch '^(?:[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+[\\/])' -or
    ($config.backend -ne 'vllm' -or [IO.Path]::GetFileName($config.executable) -ine 'python.exe') -or
    $config.command -isnot [array] -or $config.command.Count -eq 0 -or
    $config.command[0] -cne $config.executable -or !$process.Path) {
    throw 'Invalid managed server executable in config; refusing to stop it.'
}
$expectedPath = [IO.Path]::GetFullPath($config.executable)
$runningPath = [IO.Path]::GetFullPath($process.Path)
if (![String]::Equals($expectedPath, $runningPath, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'PID belongs to another executable; refusing to stop it.'
}
if ($null -ne $config.managed_pid -and $config.managed_pid -ne $serverProcessId) {
    throw 'PID does not match the managed server config; refusing to stop it.'
}
if ($null -ne $config.process_creation_filetime -and
    [string]$config.process_creation_filetime -cne [string]$process.StartTime.ToUniversalTime().ToFileTimeUtc()) {
    throw 'Process creation time differs from the managed server config; refusing to stop a reused PID.'
}
& taskkill.exe /PID $serverProcessId /T /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the owned vLLM process tree.' }
Remove-Item -LiteralPath $pidFile
Write-Output "Stopped managed server PID $serverProcessId."
