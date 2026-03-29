@echo off
setlocal
cd /d C:\Users\Frere\Downloads\AIProspection\PytorchVulkan\pytorch
echo ===== RESUME %DATE% %TIME% =====>> torch_python-build-seq.log
call "C:\Users\Frere\Downloads\AIProspection\PytorchVulkan\pytorch\vs2022-cmake.cmd" --build build --target torch_python --config Release -- /m:1 /p:BuildInParallel=false /v:minimal >> torch_python-build-seq.log 2>&1
set BUILD_EXIT=%ERRORLEVEL%
echo ===== EXITCODE %BUILD_EXIT% %DATE% %TIME% =====>> torch_python-build-seq.log
exit /b %BUILD_EXIT%
