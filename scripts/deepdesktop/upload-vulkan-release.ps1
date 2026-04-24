param(
    [string]$Repo = "laurentdupin/pytorch",
    [string]$Tag = "",
    [string]$Title = "",
    [string]$Target = "",
    [string]$WheelRoot = "dist-vulkan",
    [string[]]$PythonTags = @("cp310", "cp312", "cp314"),
    [switch]$Draft,
    [switch]$Latest,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-WorkPath {
    param(
        [string]$Requested,
        [string]$RepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($Requested)) {
        return [System.IO.Path]::GetFullPath($Requested)
    }

    return (Join-Path $RepoRoot $Requested)
}

function Invoke-Git {
    param(
        [string]$RepoRoot,
        [string[]]$Arguments
    )

    $safeRepoRoot = $RepoRoot.Replace('\', '/')
    $output = & git -c "safe.directory=$safeRepoRoot" -C $RepoRoot @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed."
    }

    return ($output | Select-Object -First 1).Trim()
}

function Get-RequiredWheel {
    param(
        [string]$Root,
        [string]$PythonTag
    )

    $wheel = Get-ChildItem -LiteralPath $Root -Recurse -Filter "*.whl" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*-$PythonTag-$PythonTag-win_amd64.whl" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $wheel) {
        throw "Missing wheel for $PythonTag under '$Root'."
    }

    return $wheel
}

function Resolve-GitHubCli {
    $command = Get-Command gh -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles "GitHub CLI\gh.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "GitHub CLI\gh.exe"),
        (Join-Path $env:LocalAppData "Programs\GitHub CLI\gh.exe")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return ""
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$version = (Get-Content -LiteralPath (Join-Path $repoRoot "version.txt") -First 1).Trim()
$shortSha = Invoke-Git -RepoRoot $repoRoot -Arguments @("rev-parse", "--short=7", "HEAD")
if (-not $Target) {
    $Target = Invoke-Git -RepoRoot $repoRoot -Arguments @("rev-parse", "HEAD")
}
if (-not $Tag) {
    $Tag = "vulkan-backend-torch-$version-git$shortSha"
}
if (-not $Title) {
    $Title = "Vulkan backend torch $version git$shortSha"
}

$wheelRootPath = Resolve-WorkPath -Requested $WheelRoot -RepoRoot $repoRoot
$wheels = foreach ($pythonTag in $PythonTags) {
    Get-RequiredWheel -Root $wheelRootPath -PythonTag $pythonTag
}

$bodyLines = @(
    "DeepDesktop Vulkan backend PyTorch wheels.",
    "",
    "Commit: $Target",
    "Version: $version",
    "",
    "Wheels:",
    ($wheels | ForEach-Object { "- $($_.Name)" })
)
$body = $bodyLines -join "`n"

Write-Host "Repo       : $Repo"
Write-Host "Tag        : $Tag"
Write-Host "Title      : $Title"
Write-Host "Target     : $Target"
Write-Host "Wheel root : $wheelRootPath"
Write-Host "Draft      : $([bool]$Draft)"
Write-Host "Prerelease : true"
Write-Host "Latest     : $([bool]$Latest)"
foreach ($wheel in $wheels) {
    Write-Host "Asset      : $($wheel.FullName) ($($wheel.Length) bytes)"
}

if ($DryRun) {
    Write-Host "Dry run only. Release was not created."
    exit 0
}

$ghPath = Resolve-GitHubCli
if (-not $ghPath) {
    throw "GitHub CLI 'gh' was not found. Install it, then run 'gh auth login'."
}

& $ghPath auth status --hostname github.com | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run 'gh auth login'."
}

$releaseArgs = @(
    "release", "create", $Tag,
    "--repo", $Repo,
    "--target", $Target,
    "--title", $Title,
    "--notes", $body,
    "--prerelease"
)

if ($Draft) {
    $releaseArgs += "--draft"
}

if ($Latest) {
    $releaseArgs += "--latest"
} else {
    $releaseArgs += "--latest=false"
}

$releaseArgs += ($wheels | ForEach-Object { $_.FullName })

& $ghPath @releaseArgs
if ($LASTEXITCODE -ne 0) {
    throw "GitHub release upload failed."
}

Write-Host "Release created: https://github.com/$Repo/releases/tag/$Tag"
