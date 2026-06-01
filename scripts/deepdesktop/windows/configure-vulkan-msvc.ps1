param(
    [string]$PythonExe = "",
    [string]$VulkanSdk = "",
    [string]$LibuvRoot = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}

function Resolve-PythonPath {
    param(
        [string]$Requested,
        [string]$RepoRoot
    )

    $candidates = @()
    if ($Requested) {
        $candidates += $Requested
    }

    $candidates += (Join-Path $RepoRoot "agent_space\venvs\transformers\Scripts\python.exe")
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

function Resolve-VulkanSdkPath {
    param([string]$Requested)

    $inputs = @()
    if ($Requested) {
        $inputs += $Requested
    }
    if ($env:VK_SDK_PATH) {
        $inputs += $env:VK_SDK_PATH
    }
    if ($env:VULKAN_SDK) {
        $inputs += $env:VULKAN_SDK
    }

    $knownRoot = "C:\Users\Frere\Downloads\AIProspection\PytorchVulkan\VulkanSDK"
    if (Test-Path -LiteralPath $knownRoot) {
        $inputs += $knownRoot
    }

    foreach ($inputPath in $inputs) {
        if (-not $inputPath -or -not (Test-Path -LiteralPath $inputPath)) {
            continue
        }

        $resolved = (Resolve-Path -LiteralPath $inputPath).Path
        $candidatePaths = @($resolved)
        $candidatePaths += Get-ChildItem -LiteralPath $resolved -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { $_.FullName }

        foreach ($candidate in $candidatePaths) {
            $glslc = Join-Path $candidate "Bin\glslc.exe"
            $vulkanLib = Join-Path $candidate "Lib\vulkan-1.lib"
            $vulkanHeader = Join-Path $candidate "Include\vulkan\vulkan_core.h"
            if ((Test-Path -LiteralPath $glslc) -and
                (Test-Path -LiteralPath $vulkanLib) -and
                (Test-Path -LiteralPath $vulkanHeader)) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }
        }
    }

    throw "Unable to resolve a complete Vulkan SDK. Pass -VulkanSdk explicitly."
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

function Remove-BuildDirectory {
    param([string]$RepoRoot)

    $buildDir = Join-Path $RepoRoot "build"
    if (-not (Test-Path -LiteralPath $buildDir)) {
        return
    }

    $resolved = (Resolve-Path -LiteralPath $buildDir).Path
    if (-not $resolved.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside repo: $resolved"
    }

    Remove-Item -LiteralPath $resolved -Recurse -Force
}

$repoRoot = Resolve-RepoRoot
$pythonPath = Resolve-PythonPath -Requested $PythonExe -RepoRoot $repoRoot
$vulkanSdkPath = Resolve-VulkanSdkPath -Requested $VulkanSdk
$libuvRootPath = Resolve-LibuvRoot -Requested $LibuvRoot -RepoRoot $repoRoot

if ($Clean) {
    Remove-BuildDirectory -RepoRoot $repoRoot
}

$env:VULKAN_SDK = $vulkanSdkPath
$env:VK_SDK_PATH = $vulkanSdkPath
$env:libuv_ROOT = $libuvRootPath
$env:PYTHONPATH = $repoRoot

$vulkanLib = Join-Path $vulkanSdkPath "Lib\vulkan-1.lib"
$vulkanInclude = Join-Path $vulkanSdkPath "Include"

Write-Host "Repo root    : $repoRoot"
Write-Host "Python       : $pythonPath"
Write-Host "Vulkan SDK   : $vulkanSdkPath"
Write-Host "libuv root   : $libuvRootPath"
Write-Host "Generator    : Visual Studio 17 2022"
Write-Host "Platform     : x64"
Write-Host "Toolset      : host=x64"

cmake -S $repoRoot -B (Join-Path $repoRoot "build") `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -T host=x64 `
    -DPython_EXECUTABLE="$pythonPath" `
    -DVulkan_LIBRARY="$vulkanLib" `
    -DVulkan_INCLUDE_DIR="$vulkanInclude" `
    -DUSE_CUDA=OFF `
    -DUSE_ROCM=OFF `
    -DUSE_MKLDNN=OFF `
    -DUSE_OPENMP=OFF `
    -DUSE_XPU=OFF `
    -DUSE_MPI=OFF `
    -DUSE_VULKAN=ON `
    -DUSE_VULKAN_API=ON `
    -DUSE_DISTRIBUTED=ON `
    -DUSE_GLOO=ON `
    -DUSE_TENSORPIPE=OFF `
    -DBUILD_TEST=ON

exit $LASTEXITCODE
