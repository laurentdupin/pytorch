param(
    [string]$PythonExe = "",
    [string]$VulkanSdk = "",
    [string]$LibuvRoot = "",
    [string]$OutDir = "dist-vulkan",
    [string]$VenvDir = "",
    [string]$CMakeGenerator = "",
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
$DefaultMaxJobs = 4

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
    if ($env:VK_SDK_PATH) {
        $inputs += $env:VK_SDK_PATH
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
        $vulkanLib = Join-Path $resolved "Lib\vulkan-1.lib"
        if ((Test-Path -LiteralPath $glslc) -and (Test-Path -LiteralPath $vulkanLib)) {
            return $resolved
        }

        $versioned = Get-ChildItem -LiteralPath $resolved -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                (Test-Path -LiteralPath (Join-Path $_.FullName "Bin\glslc.exe")) -and
                (Test-Path -LiteralPath (Join-Path $_.FullName "Lib\vulkan-1.lib"))
            } |
            Sort-Object Name -Descending |
            Select-Object -First 1

        if ($null -ne $versioned) {
            return $versioned.FullName
        }
    }

    throw "Unable to resolve a Vulkan SDK directory containing Bin\glslc.exe. Pass -VulkanSdk explicitly."
}

function Resolve-LibuvRoot {
    param(
        [string]$Requested,
        [string]$RepoRoot
    )

    $inputs = @()
    if ($Requested) {
        $inputs += $Requested
    }
    if ($env:libuv_ROOT) {
        $inputs += $env:libuv_ROOT
    }
    $inputs += (Join-Path $RepoRoot "agent_space\libuv_install")

    foreach ($inputPath in $inputs) {
        if (-not $inputPath -or -not (Test-Path -LiteralPath $inputPath)) {
            continue
        }

        $resolved = (Resolve-Path -LiteralPath $inputPath).Path
        $uvLib = Join-Path $resolved "lib\uv.lib"
        $uvHeader = Join-Path $resolved "include\uv.h"
        if ((Test-Path -LiteralPath $uvLib) -and (Test-Path -LiteralPath $uvHeader)) {
            return $resolved
        }
    }

    throw "Unable to resolve a libuv install. Pass -LibuvRoot explicitly."
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

function Get-CMakeCacheValue {
    param(
        [string]$RepoRoot,
        [string]$Name
    )

    $cachePath = Join-Path $RepoRoot "build\CMakeCache.txt"
    if (-not (Test-Path -LiteralPath $cachePath)) {
        return ""
    }

    $line = Select-String -LiteralPath $cachePath -Pattern ("^{0}:" -f [regex]::Escape($Name)) -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $line) {
        return ""
    }

    $parts = $line.Line -split '=', 2
    if ($parts.Count -ne 2) {
        return ""
    }

    return $parts[1].Trim()
}

function Normalize-ComparablePath {
    param([string]$Value)

    if (-not $Value) {
        return ""
    }

    $candidate = $Value.Replace('/', '\')
    try {
        return [System.IO.Path]::GetFullPath($candidate).TrimEnd('\').ToLowerInvariant()
    } catch {
        return $candidate.TrimEnd('\').ToLowerInvariant()
    }
}

function Resolve-CMakeGenerator {
    param(
        [string]$Requested,
        [string]$RepoRoot,
        [switch]$CleanBuild
    )

    if ($Requested) {
        return $Requested
    }

    if ($CleanBuild) {
        return "Visual Studio 17 2022"
    }

    $cachePath = Join-Path $RepoRoot "build\CMakeCache.txt"
    if (-not (Test-Path -LiteralPath $cachePath)) {
        return "Visual Studio 17 2022"
    }

    $line = Select-String -LiteralPath $cachePath -Pattern '^CMAKE_GENERATOR:INTERNAL=' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $line) {
        return "Visual Studio 17 2022"
    }

    $value = $line.Line.Substring("CMAKE_GENERATOR:INTERNAL=".Length).Trim()
    if (-not $value) {
        return "Visual Studio 17 2022"
    }

    return $value
}

function Resolve-CMakeRefresh {
    param(
        [string]$RepoRoot,
        [string]$RequestedPythonPath,
        [switch]$CleanBuild
    )

    if ($CleanBuild) {
        return [pscustomobject]@{
            Enabled = $false
            Reason = "clean build"
        }
    }

    $cachedPython = Get-CMakeCacheValue -RepoRoot $RepoRoot -Name "Python_EXECUTABLE"
    if (-not $cachedPython) {
        $cachedPython = Get-CMakeCacheValue -RepoRoot $RepoRoot -Name "_Python_EXECUTABLE"
    }
    if (-not $cachedPython) {
        $cachedPython = Get-CMakeCacheValue -RepoRoot $RepoRoot -Name "_Python3_EXECUTABLE"
    }

    if (-not $cachedPython) {
        return [pscustomobject]@{
            Enabled = $false
            Reason = "no cache"
        }
    }

    $normalizedCached = Normalize-ComparablePath -Value $cachedPython
    $normalizedRequested = Normalize-ComparablePath -Value $RequestedPythonPath
    if ($normalizedCached -eq $normalizedRequested) {
        return [pscustomobject]@{
            Enabled = $false
            Reason = "python match"
        }
    }

    return [pscustomobject]@{
        Enabled = $true
        Reason = "python mismatch"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$basePythonPath = Resolve-PythonPath -Requested $PythonExe -RepoRoot $repoRoot
$buildPython = Ensure-BuildPython -BasePythonPath $basePythonPath -RepoRoot $repoRoot -RequestedVenvDir $VenvDir -DisableVenv:$NoVenv -ResetVenv:$CleanVenv -DryRunMode:$DryRun
$pythonPath = $buildPython.BuildPythonPath
$toolsDir = $buildPython.ToolsDir
$vulkanSdkPath = Resolve-VulkanSdkPath -Requested $VulkanSdk
$libuvRootPath = Resolve-LibuvRoot -Requested $LibuvRoot -RepoRoot $repoRoot
$vsDevCmd = Resolve-VsDevCmd
$cmakeGenerator = Resolve-CMakeGenerator -Requested $CMakeGenerator -RepoRoot $repoRoot -CleanBuild:$Clean
$cmakeRefresh = Resolve-CMakeRefresh -RepoRoot $repoRoot -RequestedPythonPath $pythonPath -CleanBuild:$Clean

if (-not $DryRun -or (-not $buildPython.PendingBootstrap -and -not $buildPython.PendingCreate)) {
    if (-not (Test-CommandExists -Name "cmake" -ToolsDir $toolsDir)) {
        throw "cmake was not found for the selected build environment."
    }

    if ($cmakeGenerator -eq "Ninja" -and -not (Test-CommandExists -Name "ninja" -ToolsDir $toolsDir)) {
        throw "ninja was not found for the selected build environment."
    }
}

$glslcPath = Join-Path $vulkanSdkPath "Bin\glslc.exe"
if (-not (Test-Path -LiteralPath $glslcPath)) {
    throw "glslc.exe was not found at '$glslcPath'."
}

if ($MaxJobs -le 0) {
    $autoMaxJobs = [Math]::Max([Environment]::ProcessorCount - 1, 1)
    $MaxJobs = [Math]::Min($autoMaxJobs, $DefaultMaxJobs)
}

$outDirPath = Resolve-WorkPath -Requested $OutDir -RepoRoot $repoRoot
if (-not $outDirPath) {
    throw "Unable to resolve an output directory."
}

$existingPostCl = $env:_CL_
if ($existingPostCl) {
    $postClFlags = "$existingPostCl /MP1"
} else {
    $postClFlags = "/MP1"
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
    "set `"VK_SDK_PATH=$vulkanSdkPath`"",
    "set `"libuv_ROOT=$libuvRootPath`"",
    "set `"Path=$pathPrefix;%Path%`"",
    "set `"CMAKE_GENERATOR=$cmakeGenerator`"",
    "set `"CMAKE_BUILD_TYPE=Release`"",
    "set `"CMAKE_BUILD_PARALLEL_LEVEL=$MaxJobs`"",
    "set `"MAX_JOBS=$MaxJobs`"",
    "set `"_CL_=$postClFlags`"",
    "set `"USE_VULKAN=1`"",
    "set `"USE_VULKAN_FP16_INFERENCE=$fp16Flag`"",
    "set `"USE_VULKAN_RELAXED_PRECISION=$relaxedPrecisionFlag`"",
    "set `"USE_CUDA=0`"",
    "set `"USE_ROCM=0`"",
    "set `"USE_DISTRIBUTED=1`"",
    "set `"USE_GLOO=1`"",
    "set `"USE_C10D_GLOO=1`"",
    "set `"USE_LIBUV=1`"",
    "set `"USE_MPI=0`"",
    "set `"USE_C10D_MPI=0`"",
    "set `"USE_NCCL=0`"",
    "set `"USE_C10D_NCCL=0`"",
    "set `"USE_TENSORPIPE=0`"",
    "set `"USE_XPU=0`"",
    "set `"BUILD_TEST=0`"",
    "set `"BUILD_BINARY=0`""
)

if ($cmakeGenerator -like "Visual Studio*") {
    $cmdLines += "set `"CMAKE_GENERATOR_PLATFORM=x64`""
    $cmdLines += "set `"CMAKE_GENERATOR_TOOLSET=host=x64`""
}

if ($BuildVersion) {
    $cmdLines += "set `"PYTORCH_BUILD_VERSION=$BuildVersion`""
    $cmdLines += "set `"PYTORCH_BUILD_NUMBER=$BuildNumber`""
}

if ($cmakeRefresh.Enabled) {
    $cmdLines += "set `"CMAKE_FRESH=1`""
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
    Write-Host "libuv root    : $libuvRootPath"
    Write-Host "glslc         : $glslcPath"
    Write-Host "Generator     : $cmakeGenerator"
    Write-Host "CMake refresh : $($cmakeRefresh.Reason)"
    Write-Host "Output dir    : $outDirPath"
    Write-Host "Max jobs      : $MaxJobs"
    Write-Host "CMake jobs    : $MaxJobs"
    Write-Host "CL postfix    : $postClFlags"
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
