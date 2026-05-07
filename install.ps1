# muninn installer (Windows / PowerShell).
#
# Usage:
#   irm https://raw.githubusercontent.com/vshvedov/muninn/main/install.ps1 | iex
#
# Or, after cloning the repo manually:
#   pwsh -File install.ps1
#
# Environment variables (all optional):
#   MUNINN_REPO    git URL to install from (default: vshvedov/muninn)
#   MUNINN_BRANCH  branch / tag to install (default: main)
#
# What it does:
#   1. installs uv (https://astral.sh/uv) if missing
#   2. installs muninn into an isolated uv-managed tool environment via
#      `uv tool install`, exposing the `muninn` command on PATH
#   3. warns if uv's bin dir is not on PATH
#
# Update later:
#   muninn update
#
# Uninstall:
#   uv tool uninstall muninn

$ErrorActionPreference = "Stop"

$Repo   = if ($env:MUNINN_REPO)   { $env:MUNINN_REPO }   else { "https://github.com/vshvedov/muninn.git" }
$Branch = if ($env:MUNINN_BRANCH) { $env:MUNINN_BRANCH } else { "main" }

function Write-Ok($msg)   { Write-Host "[OK]  $msg" }
function Write-Info($msg) { Write-Host "      $msg" }
function Write-Fail($msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    exit 1
}

Write-Host "Installing muninn..."
Write-Host "  os:       windows"
Write-Host "  source:   $Repo ($Branch)"
Write-Host ""

# ---------- pre-flight ------------------------------------------------

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail @"
git not found (uv needs it to clone the muninn source). Install git first:
  winget install --id Git.Git -e
  # or download from https://git-scm.com/download/win
"@
}
Write-Ok ("git " + ((git --version) -split ' ')[2])

# ---------- uv --------------------------------------------------------
# uv brings its own managed Python interpreter; the user does NOT need
# Python installed. If uv is missing, install it via Astral's official
# PowerShell bootstrapper.

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info "uv not found, installing via https://astral.sh/uv/install.ps1 ..."
    # Download the installer to a string and only Invoke-Expression on success.
    # If Invoke-RestMethod fails partway (DNS, network blip, captive portal),
    # we surface a concrete error here instead of letting Invoke-Expression
    # evaluate truncated content and producing a misleading downstream error.
    try {
        $UvInstaller = Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1 -ErrorAction Stop
    } catch {
        Write-Fail "Could not download the uv installer from https://astral.sh/uv/install.ps1: $($_.Exception.Message). Check your network / DNS / proxy and re-run this script."
    }
    $UvInstaller | Invoke-Expression

    # uv's installer puts uv.exe under %USERPROFILE%\.local\bin and updates
    # the user PATH for new shells. Prepend that dir to the current process
    # PATH so this script can use it immediately.
    $UvBinFallback = Join-Path $HOME ".local\bin"
    if (Test-Path (Join-Path $UvBinFallback "uv.exe")) {
        $env:PATH = "$UvBinFallback;$env:PATH"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Fail "uv was installed but is not on PATH. Open a new shell and re-run this script."
}
Write-Ok ("uv " + ((uv --version) -split ' ')[1])

# ---------- install muninn -------------------------------------------
# `--reinstall` ensures re-running the installer always pulls the latest
# commit on $Branch (uv caches git sources by URL+ref otherwise).

Write-Info "Installing muninn from $Repo@$Branch ..."
& uv tool install --reinstall --from "git+$Repo@$Branch" muninn
if ($LASTEXITCODE -ne 0) {
    Write-Fail "uv tool install failed (exit $LASTEXITCODE)."
}
Write-Ok "muninn installed"

# ---------- PATH check ------------------------------------------------
# `uv tool install` places binaries in `uv tool dir --bin` (typically
# %USERPROFILE%\.local\bin). Warn the user if that dir is not on PATH.

$UvBinDir = (& uv tool dir --bin) 2>$null
if (-not $UvBinDir) { $UvBinDir = Join-Path $HOME ".local\bin" }
$UvBinDir = $UvBinDir.Trim()

$PathDirs = $env:PATH -split ';'
if ($PathDirs -notcontains $UvBinDir) {
    # Print a PowerShell-native one-liner instead of `setx`. `setx` works in
    # both cmd.exe and PowerShell but evaluates the value as a literal string,
    # so a `setx PATH "$dir;$env:PATH"` line copy-pasted into cmd.exe would
    # store the literal text `$env:PATH` into the User PATH and break it.
    # [Environment]::SetEnvironmentVariable resolves the current user PATH
    # safely from inside PowerShell only.
    Write-Host ""
    Write-Host "[WARN] $UvBinDir is not on your PATH." -ForegroundColor Yellow
    Write-Host "       Run this in PowerShell to add it permanently:" -ForegroundColor Yellow
    Write-Host "           [Environment]::SetEnvironmentVariable('PATH', '$UvBinDir;' + [Environment]::GetEnvironmentVariable('PATH','User'), 'User')" -ForegroundColor Yellow
    Write-Host "       Then open a new shell." -ForegroundColor Yellow
}

# ---------- next steps ------------------------------------------------

Write-Host ""
Write-Host "[OK]  muninn installed"
Write-Host "   command:  $UvBinDir\muninn.exe"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Make sure Ollama is running:   ollama serve"
Write-Host "  2. Pull a compatible model:       ollama pull qwen3-coder:30b"
Write-Host "  3. Run in any project directory:  muninn C:\path\to\project"
Write-Host ""
Write-Host "To upgrade later:    muninn update"
Write-Host "To uninstall:        uv tool uninstall muninn"
