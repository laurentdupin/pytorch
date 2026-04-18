param(
    [string]$PythonExe = "",
    [string]$VulkanSdk = "",
    [string]$OutDir = "dist-vulkan",
    [string]$VenvDir = "",
    [int]$MaxJobs = 0,
    [switch]$Clean,
    [switch]$CleanVenv,
    [switch]$DryRun,
    [switch]$NoVenv,
    [switch]$DisableFp16,
    [switch]$RelaxedPrecision,
    [string]$BuildVersion = "",
    [int]$BuildNumber = 1
)

$ErrorActionPreference = "Stop"

function Resolve-PythonPath {
    param(
        [string]$Requested,
        [string]$RepoRoot
    )

    $candidates = @()

    if ($Requested) {
        $candidates += $Requested
    }

    $candidates += (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    $candidates += "python.exe"
    $candidates += "python"

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }

        if ([System.IO.Path]::IsPathRooted($candidate)) {
            if (Test-Path -LiteralPath $candidate) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }

            continue
        }

        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command.Source
        }
    }

    throw "Unable to find a Python interpreter. Pass -PythonExe explicitly."
}

function Resolve-WorkPath {
    param(
        [string]$Requested,
        [string]$RepoRoot
    )

    if (-not $Requested) {
        return ""
    }

    if ([System.IO.Path]::IsPathRooted($Requested)) {
        return [System.IO.Path]::GetFullPath($Requested)
    }

    return (Join-Path $RepoRoot $Requested)
}

function Get-PythonVersionInfo {
    param([string]$PythonPath)

    $output = & $PythonPath -c "import sys; print(f'py{sys.version_info[0]}{sys.version_info[1]}'); print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
    if ($LASTEXITCODE -ne 0 -or $null -eq $output -or $output.Count -lt 2) {
        throw "Unable to determine the Python version for '$PythonPath'."
    }

    return [pscustomobject]@{
        Tag = $output[0].Trim()
        Version = $output[1].Trim()
    }
}

function Get-BuildVenvState {
    param(
        [string]$RepoRoot,
        [string]$BasePythonPath
    )

    $requirementsPath = Join-Path $RepoRoot "requirements-build.txt"
    if (-not (Test-Path -LiteralPath $requirementsPath)) {
        throw "Unable to locate '$requirementsPath'."
    }

    return [pscustomobject]@{
        StateVersion = 1
        BasePython = $BasePythonPath
        RequirementsPath = $requirementsPath
        RequirementsHash = (Get-FileHash -LiteralPath $requirementsPath -Algorithm SHA256).Hash
    }
}

function Test-BuildPythonEnvironment {
    param([string]$PythonPath)

    & $PythonPath -c "import build, cmake, ninja, numpy, yaml, requests, packaging, six, typing_extensions" | Out-Null
    return $LASTEXITCODE -eq 0
}

function Ensure-BuildPython {
    param(
        [string]$BasePythonPath,
        [string]$RepoRoot,
        [string]$RequestedVenvDir,
        [switch]$DisableVenv,
        [switch]$ResetVenv,
        [switch]$DryRunMode
    )

    $baseVersionInfo = Get-PythonVersionInfo -PythonPath $BasePythonPath

    if ($DisableVenv) {
        & $BasePythonPath -m build --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Python build frontend is missing for '$BasePythonPath'. Install the 'build' package in that environment."
        }

        return [pscustomobject]@{
            BasePythonPath = $BasePythonPath
            BuildPythonPath = $BasePythonPath
            ToolsDir = Split-Path -Parent $BasePythonPath
            VenvDir = ""
            UsedVenv = $false
            PendingCreate = $false
            PendingBootstrap = $false
            StateReason = "disabled"
            VersionTag = $baseVersionInfo.Tag
            VersionString = $baseVersionInfo.Version
        }
    }

    $resolvedVenvDir = Resolve-WorkPath -Requested $RequestedVenvDir -RepoRoot $RepoRoot
    if (-not $resolvedVenvDir) {
        $resolvedVenvDir = Join-Path $RepoRoot (Join-Path ".build-venvs" $baseVersionInfo.Tag)
    }

    $venvPythonPath = Join-Path $resolvedVenvDir "Scripts\python.exe"
    $venvToolsDir = Join-Path $resolvedVenvDir "Scripts"
    $statePath = Join-Path $resolvedVenvDir ".deepdesktop-build-state.json"
    $expectedState = Get-BuildVenvState -RepoRoot $RepoRoot -BasePythonPath $BasePythonPath

    $pendingCreate = $ResetVenv -or -not (Test-Path -LiteralPath $venvPythonPath)
    $pendingBootstrap = $pendingCreate
    $stateReason = if ($pendingCreate) { "missing or reset" } else { "ready" }

    if (-not $pendingBootstrap) {
        $needsBootstrap = $false

        if (-not (Test-Path -LiteralPath $statePath)) {
            if (Test-BuildPythonEnvironment -PythonPath $venvPythonPath) {
                $expectedState | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding Ascii
                $stateReason = "reused"
            } else {
                $needsBootstrap = $true
                $stateReason = "missing state"
            }
        } else {
            try {
                $currentState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                if ($currentState.StateVersion -ne $expectedState.StateVersion -or
                    $currentState.BasePython -ne $expectedState.BasePython -or
                    $currentState.RequirementsHash -ne $expectedState.RequirementsHash) {
                    $needsBootstrap = $true
                    $stateReason = "state mismatch"
                }
            } catch {
                $needsBootstrap = $true
                $stateReason = "invalid state"
            }
        }

        if (-not $needsBootstrap) {
            & $venvPythonPath -m build --version | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $needsBootstrap = $true
                $stateReason = "missing build frontend"
            }
        }

        $pendingBootstrap = $needsBootstrap
    }

    if ($DryRunMode) {
        return [pscustomobject]@{
            BasePythonPath = $BasePythonPath
            BuildPythonPath = $venvPythonPath
            ToolsDir = $venvToolsDir
            VenvDir = $resolvedVenvDir
            UsedVenv = $true
            PendingCreate = $pendingCreate
            PendingBootstrap = $pendingBootstrap
            StateReason = $stateReason
            VersionTag = $baseVersionInfo.Tag
            VersionString = $baseVersionInfo.Version
        }
    }

    if ($ResetVenv -and (Test-Path -LiteralPath $resolvedVenvDir)) {
        Remove-Item -LiteralPath $resolvedVenvDir -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $venvPythonPath)) {
        New-Item -ItemType Directory -Force -Path $resolvedVenvDir | Out-Null
        & $BasePythonPath -m venv $resolvedVenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create build virtual environment at '$resolvedVenvDir'."
        }
    }

    if ($pendingBootstrap) {
        & $venvPythonPath -m pip install --upgrade -r $expectedState.RequirementsPath build wheel
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install build dependencies into '$resolvedVenvDir'."
        }

        & $venvPythonPath -m build --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Python build frontend is still missing in '$resolvedVenvDir' after bootstrapping."
        }

        $expectedState | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding Ascii
    }

    return [pscustomobject]@{
        BasePythonPath = $BasePythonPath
        BuildPythonPath = $venvPythonPath
        ToolsDir = $venvToolsDir
        VenvDir = $resolvedVenvDir
        UsedVenv = $true
        PendingCreate = $false
        PendingBootstrap = $false
        StateReason = if ($pendingBootstrap) { "bootstrapped" } else { "reused" }
        VersionTag = $baseVersionInfo.Tag
        VersionString = $baseVersionInfo.Version
    }
}

function Resolve-VulkanSdkPath {
    param([string]$Requested)

    $inputs = @()

    if ($Requested) {
        $inputs += $Requested
    }

    if ($env:VULKAN_SDK) {
        $inputs += $env:VULKAN_SDK
    }

    foreach ($inputPath in $inputs) {
        if (-not $inputPath) {
            continue
        }

        if (-not (Test-Path -LiteralPath $inputPath)) {
            continue
        }

        $resolved = (Resolve-Path -LiteralPath $inputPath).Path
        $glslc = Join-Path $resolved "Bin\glslc.exe"
        if (Test-Path -LiteralPath $glslc) {
            return $resolved
        }

        $versioned = Get-ChildItem -LiteralPath $resolved -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "Bin\glslc.exe") } |
            Sort-Object Name -Descending |
            Select-Object -First 1

        if ($null -ne $versioned) {
            return $versioned.FullName
        }
    }

    throw "Unable to resolve a Vulkan SDK directory containing Bin\glslc.exe. Pass -VulkanSdk explicitly."
}

function Resolve-VsDevCmd {
    $vsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vsWhere)) {
        throw "Unable to locate vswhere.exe."
    }

    $installationPath = & $vsWhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
    if (-not $installationPath) {
        throw "Unable to locate a Visual Studio installation with MSBuild."
    }

    $vsDevCmd = Join-Path $installationPath.Trim() "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd)) {
        throw "Unable to locate VsDevCmd.bat at '$vsDevCmd'."
    }

    return $vsDevCmd
}

function Test-CommandExists {
    param(
        [string]$Name,
        [string]$ToolsDir = ""
    )

    if ($ToolsDir) {
        $candidate = Join-Path $ToolsDir "$Name.exe"
        if (Test-Path -LiteralPath $candidate) {
            return $true
        }
    }

    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$basePythonPath = Resolve-PythonPath -Requested $PythonExe -RepoRoot $repoRoot
$buildPython = Ensure-BuildPython -BasePythonPath $basePythonPath -RepoRoot $repoRoot -RequestedVenvDir $VenvDir -DisableVenv:$NoVenv -ResetVenv:$CleanVenv -DryRunMode:$DryRun
$pythonPath = $buildPython.BuildPythonPath
$toolsDir = $buildPython.ToolsDir
$vulkanSdkPath = Resolve-VulkanSdkPath -Requested $VulkanSdk
$vsDevCmd = Resolve-VsDevCmd

if (-not $DryRun -or (-not $buildPython.PendingBootstrap -and -not $buildPython.PendingCreate)) {
    if (-not (Test-CommandExists -Name "cmake" -ToolsDir $toolsDir)) {
        throw "cmake was not found for the selected build environment."
    }

    if (-not (Test-CommandExists -Name "ninja" -ToolsDir $toolsDir)) {
        throw "ninja was not found for the selected build environment."
    }
}

$glslcPath = Join-Path $vulkanSdkPath "Bin\glslc.exe"
if (-not (Test-Path -LiteralPath $glslcPath)) {
    throw "glslc.exe was not found at '$glslcPath'."
}

if ($MaxJobs -le 0) {
    $MaxJobs = [Math]::Max([Environment]::ProcessorCount - 1, 1)
}

$outDirPath = Resolve-WorkPath -Requested $OutDir -RepoRoot $repoRoot
if (-not $outDirPath) {
    throw "Unable to resolve an output directory."
}

if ($Clean) {
    $buildDir = Join-Path $repoRoot "build"
    if (Test-Path -LiteralPath $buildDir) {
        Remove-Item -LiteralPath $buildDir -Recurse -Force
    }

    if (Test-Path -LiteralPath $outDirPath) {
        Remove-Item -LiteralPath $outDirPath -Recurse -Force
    }
}

New-Item -ItemType Directory -Force -Path $outDirPath | Out-Null

$cmdFile = Join-Path $env:TEMP ("deepdesktop-vulkan-wheel-{0}.cmd" -f $PID)
$fp16Flag = if ($DisableFp16) { "0" } else { "1" }
$relaxedPrecisionFlag = if ($RelaxedPrecision) { "1" } else { "0" }
$pathPrefix = if ($buildPython.UsedVenv) {
    "$toolsDir;$vulkanSdkPath\Bin"
} else {
    "$vulkanSdkPath\Bin"
}

$cmdLines = @(
    "@echo off",
    "setlocal",
    "call `"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul",
    "if errorlevel 1 exit /b %errorlevel%",
    "cd /d `"$repoRoot`"",
    "set `"VULKAN_SDK=$vulkanSdkPath`"",
    "set `"PATH=$pathPrefix;%PATH%`"",
    "set `"CMAKE_GENERATOR=Ninja`"",
    "set `"CMAKE_BUILD_TYPE=Release`"",
    "set `"MAX_JOBS=$MaxJobs`"",
    "set `"USE_VULKAN=1`"",
    "set `"USE_VULKAN_FP16_INFERENCE=$fp16Flag`"",
    "set `"USE_VULKAN_RELAXED_PRECISION=$relaxedPrecisionFlag`"",
    "set `"USE_CUDA=0`"",
    "set `"USE_ROCM=0`"",
    "set `"USE_DISTRIBUTED=0`"",
    "set `"USE_GLOO=0`"",
    "set `"USE_MPI=0`"",
    "set `"USE_TENSORPIPE=0`"",
    "set `"USE_XPU=0`"",
    "set `"BUILD_TEST=0`"",
    "set `"BUILD_BINARY=0`""
)

if ($BuildVersion) {
    $cmdLines += "set `"PYTORCH_BUILD_VERSION=$BuildVersion`""
    $cmdLines += "set `"PYTORCH_BUILD_NUMBER=$BuildNumber`""
}

$cmdLines += "`"$pythonPath`" -m build --wheel --no-isolation --outdir `"$outDirPath`""
$cmdLines += "exit /b %errorlevel%"

Set-Content -LiteralPath $cmdFile -Encoding Ascii -Value $cmdLines

try {
    Write-Host "Repo root     : $repoRoot"
    Write-Host "Base Python   : $basePythonPath"
    Write-Host "Build Python  : $pythonPath"
    Write-Host "Python ver.   : $($buildPython.VersionString)"
    if ($buildPython.UsedVenv) {
        Write-Host "Build venv    : $($buildPython.VenvDir)"
        if ($DryRun -and ($buildPython.PendingCreate -or $buildPython.PendingBootstrap)) {
            Write-Host "Venv status   : would prepare ($($buildPython.StateReason))"
        } else {
            Write-Host "Venv status   : $($buildPython.StateReason)"
        }
    } else {
        Write-Host "Build venv    : disabled"
    }
    Write-Host "Vulkan SDK    : $vulkanSdkPath"
    Write-Host "glslc         : $glslcPath"
    Write-Host "Output dir    : $outDirPath"
    Write-Host "Max jobs      : $MaxJobs"
    Write-Host "FP16 shaders  : $fp16Flag"
    Write-Host "Relaxed prec. : $relaxedPrecisionFlag"

    if ($DryRun) {
        Write-Host "Dry run only. Wheel build was not executed."
        $exitCode = 0
        return
    }

    & cmd.exe /d /c "`"$cmdFile`""
    $exitCode = $LASTEXITCODE
} finally {
    if (Test-Path -LiteralPath $cmdFile) {
        Remove-Item -LiteralPath $cmdFile -Force
    }
}

if ($exitCode -ne 0) {
    exit $exitCode
}

$wheel = Get-ChildItem -LiteralPath $outDirPath -Filter *.whl -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $wheel) {
    throw "Build completed without producing a wheel in '$outDirPath'."
}

Write-Host "Wheel generated: $($wheel.FullName)"
