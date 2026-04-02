@echo off
setlocal
for %%I in ("%~dp0.") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"
echo ===== RESUME %DATE% %TIME% =====>> torch_python-build-seq.log
call "%REPO_ROOT%\vs2022-cmake.cmd" --build build --target torch_python --config Release -- /m:1 /p:BuildInParallel=false /v:minimal >> torch_python-build-seq.log 2>&1
set BUILD_EXIT=%ERRORLEVEL%
echo ===== EXITCODE %BUILD_EXIT% %DATE% %TIME% =====>> torch_python-build-seq.log
exit /b %BUILD_EXIT%
