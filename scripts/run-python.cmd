@echo off
setlocal
set "HY_PYTHON=%~dp0..\runtime\vllm\python.exe"
if not exist "%HY_PYTHON%" set "HY_PYTHON=%~dp0..\runtime\python\python.exe"
if not exist "%HY_PYTHON%" set "HY_PYTHON=%~dp0..\runtime\python\cpython-3.12-windows-x86_64-none\python.exe"
if not exist "%HY_PYTHON%" (
    echo Missing bundled Python runtime. Extract the complete package first. 1>&2
    exit /b 1
)
"%HY_PYTHON%" -E -s -X utf8 %*
exit /b %errorlevel%
