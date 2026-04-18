$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$logDir = Join-Path $repoRoot ".build-logs\windows"
$log = Join-Path $logDir "torch_python-build-seq.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $repoRoot
Add-Content -Path $log -Value ("===== POWERSHELL RESUME " + (Get-Date -Format o) + " =====")

$cmd = '"' + (Join-Path $PSScriptRoot "vs2022-cmake.cmd") + '" --build build --target torch_python --config Release -- /m:1 /p:BuildInParallel=false /v:minimal >> "' + $log + '" 2>&1'
& cmd.exe /d /s /c $cmd
$exitCode = $LASTEXITCODE

Add-Content -Path $log -Value ("===== EXITCODE " + $exitCode + " " + (Get-Date -Format o) + " =====")
exit $exitCode
