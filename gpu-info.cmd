@echo off
call "%~dp0scripts\run-python.cmd" "%~dp0scripts\gpu_config.py" %*
exit /b %errorlevel%
