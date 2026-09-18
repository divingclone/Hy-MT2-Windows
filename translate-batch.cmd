@echo off
setlocal
call "%~dp0scripts\run-python.cmd" "%~dp0scripts\translate_batch.py" %*
exit /b %errorlevel%
