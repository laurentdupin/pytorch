$ErrorActionPreference = "Stop"

$wd = "C:\Users\Frere\Downloads\AIProspection\PytorchVulkan\pytorch"
$log = Join-Path $wd "torch_python-build-seq.log"

Set-Location $wd
Add-Content -Path $log -Value ("===== POWERSHELL RESUME " + (Get-Date -Format o) + " =====")

$cmd = '"' + (Join-Path $wd "vs2022-cmake.cmd") + '" --build build --target torch_python --config Release -- /m:1 /p:BuildInParallel=false /v:minimal >> torch_python-build-seq.log 2>&1'
& cmd.exe /d /s /c $cmd
$exitCode = $LASTEXITCODE

Add-Content -Path $log -Value ("===== EXITCODE " + $exitCode + " " + (Get-Date -Format o) + " =====")
exit $exitCode
