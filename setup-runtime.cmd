@echo off
call "%~dp0scripts\run-python.cmd" "%~dp0scripts\prepare_vllm_runtime.py" %*
exit /b %errorlevel%
