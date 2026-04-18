@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-vulkan-wheel-build.ps1" %*
exit /b %errorlevel%
