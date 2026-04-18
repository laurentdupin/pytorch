@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-vulkan-wheel.ps1" %*
exit /b %errorlevel%
