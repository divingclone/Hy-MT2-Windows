@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  if exist "runtime\python\cpython-3.12-windows-x86_64-none\python.exe" (
    "runtime\python\cpython-3.12-windows-x86_64-none\python.exe" -m venv .venv
  ) else (
    py -3.12 -m venv .venv
  )
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
exit /b %errorlevel%
