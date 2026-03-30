@echo off
setlocal
set "_CODEX_TEMP=%TEMP%"
set "_CODEX_TMP=%TMP%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
if defined _CODEX_TEMP set "TEMP=%_CODEX_TEMP%"
if defined _CODEX_TMP set "TMP=%_CODEX_TMP%"
"C:\Users\Frere\Downloads\AIProspection\PytorchVulkan\.venv\Scripts\python.exe" %*
exit /b %errorlevel%
