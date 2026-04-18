param(
    [string]$VulkanSdk = "",
    [string[]]$PythonVersions = @("3.10", "3.12", "3.14"),
    [switch]$CleanVenv,
    [switch]$DryRun,
    [switch]$SkipPythonInstall,
    [switch]$SkipVulkanSdkEnv
)

$ErrorActionPreference = "Stop"

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

function Resolve-PythonInstallPaths {
    param([string]$Version)

    $suffix = $Version.Replace(".", "")
    return @(
        (Join-Path $env:ProgramFiles ("Python{0}\python.exe" -f $suffix)),
        (Join-Path $env:LocalAppData ("Programs\Python\Python{0}\python.exe" -f $suffix))
    )
}

function Resolve-PythonForVersion {
    param([string]$Version)

    if ($null -ne (Get-Command py -ErrorAction SilentlyContinue)) {
        try {
            $probe = & py -$Version -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe) {
                $resolvedProbe = $probe | Select-Object -First 1
                if (Test-Path -LiteralPath $resolvedProbe) {
                    return (Resolve-Path -LiteralPath $resolvedProbe).Path
                }
            }
        } catch {
        }
    }

    foreach ($candidate in (Resolve-PythonInstallPaths -Version $Version)) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return ""
}

function Install-PythonVersion {
    param(
        [string]$Version,
        [switch]$DryRunMode
    )

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        throw "winget.exe was not found. Install Python $Version manually or make winget available."
    }

    $packageId = "Python.Python.$Version"
    if ($DryRunMode) {
        return $packageId
    }

    & $winget.Source install --id $packageId --exact --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python $Version with winget."
    }

    Start-Sleep -Seconds 2
    return $packageId
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

function Ensure-BuildVenv {
    param(
        [string]$BasePythonPath,
        [string]$RepoRoot,
        [switch]$ResetVenv,
        [switch]$DryRunMode
    )

    $versionInfo = Get-PythonVersionInfo -PythonPath $BasePythonPath
    $resolvedVenvDir = Join-Path $RepoRoot (Join-Path ".build-venvs" $versionInfo.Tag)
    $venvPythonPath = Join-Path $resolvedVenvDir "Scripts\python.exe"
    $statePath = Join-Path $resolvedVenvDir ".deepdesktop-build-state.json"
    $expectedState = Get-BuildVenvState -RepoRoot $RepoRoot -BasePythonPath $BasePythonPath

    $pendingCreate = $ResetVenv -or -not (Test-Path -LiteralPath $venvPythonPath)
    $pendingBootstrap = $pendingCreate
    $stateReason = if ($pendingCreate) { "missing or reset" } else { "ready" }

    if (-not $pendingBootstrap) {
        if (-not (Test-Path -LiteralPath $statePath)) {
            if (Test-BuildPythonEnvironment -PythonPath $venvPythonPath) {
                $expectedState | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding Ascii
                $stateReason = "reused"
            } else {
                $pendingBootstrap = $true
                $stateReason = "missing state"
            }
        } else {
            try {
                $currentState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                if ($currentState.StateVersion -ne $expectedState.StateVersion -or
                    $currentState.BasePython -ne $expectedState.BasePython -or
                    $currentState.RequirementsHash -ne $expectedState.RequirementsHash) {
                    $pendingBootstrap = $true
                    $stateReason = "state mismatch"
                }
            } catch {
                $pendingBootstrap = $true
                $stateReason = "invalid state"
            }
        }

        if (-not $pendingBootstrap) {
            & $venvPythonPath -m build --version | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $pendingBootstrap = $true
                $stateReason = "missing build frontend"
            }
        }
    }

    if ($DryRunMode) {
        return [pscustomobject]@{
            BasePythonPath = $BasePythonPath
            BuildPythonPath = $venvPythonPath
            VenvDir = $resolvedVenvDir
            PendingCreate = $pendingCreate
            PendingBootstrap = $pendingBootstrap
            StateReason = $stateReason
            VersionTag = $versionInfo.Tag
            VersionString = $versionInfo.Version
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
        VenvDir = $resolvedVenvDir
        PendingCreate = $false
        PendingBootstrap = $false
        StateReason = if ($pendingBootstrap) { "bootstrapped" } else { "reused" }
        VersionTag = $versionInfo.Tag
        VersionString = $versionInfo.Version
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$vulkanSdkPath = Resolve-VulkanSdkPath -Requested $VulkanSdk
$pythonResults = @()

foreach ($version in $PythonVersions) {
    $pythonPath = Resolve-PythonForVersion -Version $version
    $pythonAction = "reused"

    if (-not $pythonPath) {
        if ($SkipPythonInstall) {
            throw "Python $version is missing. Re-run without -SkipPythonInstall or install it manually."
        }

        $packageId = Install-PythonVersion -Version $version -DryRunMode:$DryRun
        $pythonAction = if ($DryRun) { "would install ($packageId)" } else { "installed ($packageId)" }

        if (-not $DryRun) {
            $pythonPath = Resolve-PythonForVersion -Version $version
            if (-not $pythonPath) {
                throw "Python $version was installed but could not be resolved afterwards."
            }
        }
    }

    if (-not $pythonPath) {
        $pythonResults += [pscustomobject]@{
            RequestedVersion = $version
            PythonPath = ""
            PythonAction = $pythonAction
            VenvDir = Join-Path $repoRoot (Join-Path ".build-venvs" ("py" + $version.Replace(".", "")))
            VenvAction = "pending install"
        }
        continue
    }

    $buildVenv = Ensure-BuildVenv -BasePythonPath $pythonPath -RepoRoot $repoRoot -ResetVenv:$CleanVenv -DryRunMode:$DryRun
    $venvAction = if ($DryRun -and ($buildVenv.PendingCreate -or $buildVenv.PendingBootstrap)) {
        "would prepare ($($buildVenv.StateReason))"
    } else {
        $buildVenv.StateReason
    }

    $pythonResults += [pscustomobject]@{
        RequestedVersion = $version
        PythonPath = $pythonPath
        PythonAction = $pythonAction
        VenvDir = $buildVenv.VenvDir
        VenvAction = $venvAction
    }
}

Write-Host "Repo root        : $repoRoot"
Write-Host "Resolved SDK     : $vulkanSdkPath"
if ($SkipVulkanSdkEnv) {
    Write-Host "User VULKAN_SDK  : skipped"
} elseif ($DryRun) {
    Write-Host "User VULKAN_SDK  : would set to $vulkanSdkPath"
} else {
    [Environment]::SetEnvironmentVariable("VULKAN_SDK", $vulkanSdkPath, "User")
    $env:VULKAN_SDK = $vulkanSdkPath
    Write-Host "User VULKAN_SDK  : $vulkanSdkPath"
}

foreach ($result in $pythonResults) {
    Write-Host ""
    Write-Host "Python $($result.RequestedVersion)"
    Write-Host "  Interpreter : $($result.PythonPath)"
    Write-Host "  Python step : $($result.PythonAction)"
    Write-Host "  Venv dir    : $($result.VenvDir)"
    Write-Host "  Venv step   : $($result.VenvAction)"
}

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run only. No installations or venv changes were executed."
}
