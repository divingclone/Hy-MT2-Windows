@echo off
setlocal
if "%~1"=="" (
    call "%~dp0scripts\run-python.cmd" "%~dp0scripts\setup_model.py" --pause-on-exit
) else (
    call "%~dp0scripts\run-python.cmd" "%~dp0scripts\setup_model.py" %*
)
exit /b %errorlevel%
