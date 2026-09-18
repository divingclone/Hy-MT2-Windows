@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\serve.ps1" -Background %*
exit /b %errorlevel%
