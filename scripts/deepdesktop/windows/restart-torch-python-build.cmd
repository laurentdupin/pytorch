@echo off
setlocal
for %%I in ("%~dp0..\..\..") do set "REPO_ROOT=%%~fI"
set "LOG_DIR=%REPO_ROOT%\.build-logs\windows"
set "LOG_FILE=%LOG_DIR%\torch_python-build-seq.log"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%REPO_ROOT%"
echo ===== RESUME %DATE% %TIME% =====>> "%LOG_FILE%"
call "%~dp0vs2022-cmake.cmd" --build build --target torch_python --config Release -- /m:1 /p:BuildInParallel=false /v:minimal >> "%LOG_FILE%" 2>&1
set BUILD_EXIT=%ERRORLEVEL%
echo ===== EXITCODE %BUILD_EXIT% %DATE% %TIME% =====>> "%LOG_FILE%"
exit /b %BUILD_EXIT%
