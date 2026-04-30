# =============================================================
# Install QLoRA training as a Windows service (NSSM)
# =============================================================
# Purpose: a 12-24h training run must survive RDP disconnects, user logouts,
# and "PowerShell window accidentally closed". A Windows service runs as
# LocalSystem (or whatever user you set), independent of any login session.
#
# Usage:
#   1. Install NSSM (one-time):  choco install nssm  -OR-  download nssm.cc
#   2. Run this script as ADMIN (right-click PowerShell -> Run as Administrator)
#   3. Start training: Start-Service QwenTrain  (or via services.msc)
#   4. Tail progress: Get-Content outputs\stdout.log -Wait -Tail 50
#   5. Stop: Stop-Service QwenTrain
#   6. Uninstall: nssm remove QwenTrain confirm
#
# Heartbeat: while the service runs, training_runs table in Supabase is
# updated every ~50 steps (configurable in scripts/train_qlora.py).
# =============================================================

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ServiceName = "QwenTrain"
$VenvPython  = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$LauncherPy  = Join-Path $ProjectRoot "scripts\train_qlora.py"
$StdoutLog   = Join-Path $ProjectRoot "outputs\stdout.log"
$StderrLog   = Join-Path $ProjectRoot "outputs\stderr.log"

# Sanity checks
if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
    Write-Error "NSSM not found. Install with: choco install nssm  OR download from nssm.cc"
}
if (-not (Test-Path $VenvPython))   { Write-Error "venv python not found at $VenvPython" }
if (-not (Test-Path $LauncherPy))   { Write-Error "Launcher not found at $LauncherPy" }
if (-not (Test-Path (Split-Path $StdoutLog))) { New-Item -ItemType Directory (Split-Path $StdoutLog) | Out-Null }

# Remove existing service if present (so this script is re-runnable)
if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing $ServiceName service..."
    Stop-Service $ServiceName -ErrorAction SilentlyContinue
    & nssm remove $ServiceName confirm
}

# Install
& nssm install $ServiceName $VenvPython $LauncherPy
& nssm set $ServiceName AppDirectory $ProjectRoot
& nssm set $ServiceName AppStdout $StdoutLog
& nssm set $ServiceName AppStderr $StderrLog
& nssm set $ServiceName AppStdoutCreationDisposition 2   # 2 = append
& nssm set $ServiceName AppStderrCreationDisposition 2

# Env vars passed to the service process. NSSM expects newline-separated entries.
# Reads SUPABASE_* from .env via the launcher's tiny dotenv loader.
$envBlock = @"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
"@
& nssm set $ServiceName AppEnvironmentExtra $envBlock

# Auto-restart on crash, but throttle: at most once every 5 min, give up after 5 retries
& nssm set $ServiceName AppExit Default Restart
& nssm set $ServiceName AppRestartDelay 300000              # 5 min between restarts
& nssm set $ServiceName AppThrottle 60000                   # treat <60s runs as failures
& nssm set $ServiceName AppStopMethodSkip 0
& nssm set $ServiceName AppStopMethodConsole 30000          # 30s graceful shutdown
& nssm set $ServiceName AppKillProcessTree 1

# Run as the current user so it sees the venv + .env file paths the same way
$me = "$env:USERDOMAIN\$env:USERNAME"
& nssm set $ServiceName ObjectName $me

# Manual start by default — never auto-start at boot (we don't want training
# to launch every time the machine reboots without intent)
& nssm set $ServiceName Start SERVICE_DEMAND_START

Write-Host ""
Write-Host "Installed service: $ServiceName" -ForegroundColor Green
Write-Host "  Logs:    $StdoutLog"
Write-Host "  Errors:  $StderrLog"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Run preflight:        python scripts/preflight_dataset.py"
Write-Host "  2. Start training:       Start-Service $ServiceName"
Write-Host "  3. Tail logs:            Get-Content $StdoutLog -Wait -Tail 50"
Write-Host "  4. Stop:                 Stop-Service $ServiceName"
Write-Host "  5. Service status:       Get-Service $ServiceName"
Write-Host "  6. Uninstall:            nssm remove $ServiceName confirm"
Write-Host ""
Write-Host "Heartbeat published to Supabase 'training_runs' table every ~50 steps."
