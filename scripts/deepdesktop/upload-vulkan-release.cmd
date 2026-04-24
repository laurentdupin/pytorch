@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload-vulkan-release.ps1" %*
exit /b %errorlevel%
