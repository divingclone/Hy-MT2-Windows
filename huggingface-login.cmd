@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Maintainer environment missing. Run setup-dev.cmd first.
  pause
  exit /b 1
)
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0scripts\huggingface-login.ps1"
set "HY_LOGIN_EXIT=%errorlevel%"
exit /b %HY_LOGIN_EXIT%
