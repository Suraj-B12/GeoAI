# ============================================================
# GeoAI — clean shutdown
# ============================================================
# Stops the uvicorn server gracefully (calls /operator/stop first to drain
# any in-flight worker, then kills the process).
#
# With DB_BACKEND=postgres (in .env or the environment) the local PostgreSQL
# cluster (PG_BIN / PG_DATA) is stopped afterwards ("fast" mode: open
# transactions roll back, nothing committed is lost). -KeepDb leaves it running.
# With DB_BACKEND=supabase it is left running even when start.ps1 started it
# for the video pipeline: its store and the GIS outlive a server restart.
#
# Usage:
#   .\stop.ps1              # graceful (drains worker first); the tunnel keeps running (same URL)
#   .\stop.ps1 -Force       # immediate kill
#   .\stop.ps1 -KeepDb      # leave the local PostgreSQL running
#   .\stop.ps1 -StopTunnel  # also stop cloudflared (the next -Tunnel start gets a NEW URL)
#
# The start.ps1 watchdog loop is stopped FIRST (before 2026-11-06 it was left running and simply
# relaunched the server this script had just killed). The drain request sends the first API_KEYS
# key (it got a silent 401 once API_KEYS was set).
# ============================================================

param([switch]$Force, [switch]$KeepDb, [switch]$StopTunnel, [int]$Port = 8000)
# Pause keepalive.ps1 (scheduled task) so it does not start everything again; start.ps1 resumes it.
New-Item -ItemType Directory -Force 'C:\GeoAI\tmp' | Out-Null
Set-Content -Path 'C:\GeoAI\tmp\keepalive.off' -Value (Get-Date -Format o)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot

function Get-EnvSetting {
    # A setting from the process environment, else from .env (same order as uvicorn's --env-file).
    param([string]$Name, [string]$Default = '')
    $v = [Environment]::GetEnvironmentVariable($Name)
    if ($v) { return $v.Trim() }
    $envFile = Join-Path $ProjectRoot '.env'
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match "^\s*$Name\s*=\s*(.*?)\s*$") { return $Matches[1].Trim('"').Trim("'") }
        }
    }
    return $Default
}

if (-not $Force) {
    Write-Host "Asking worker to drain..."
    $key = ((Get-EnvSetting 'API_KEYS' '') -split ',' | ForEach-Object { $_.Trim() } |
            Where-Object { $_ } | Select-Object -First 1)
    $headers = @{}; if ($key) { $headers['X-API-Key'] = $key }
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/operator/stop" -Headers $headers `
            -Method POST -UseBasicParsing -TimeoutSec 30 | Out-Null
        Write-Host "  Drain request sent." -ForegroundColor Green
    } catch {
        Write-Host "  /operator/stop failed (server may already be down): $($_.Exception.Message)" -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 2
}

# Stop the start.ps1 watchdog loop(s) first, or they relaunch the server killed below.
# Matches PowerShell processes whose command line runs start.ps1; never this process.
$watchdogs = Get-CimInstance Win32_Process |
    Where-Object { $_.ProcessId -ne $PID -and $_.Name -match '^(powershell|pwsh)(\.exe)?$' -and
                   $_.CommandLine -and $_.CommandLine -match 'start\.ps1' }
foreach ($w in $watchdogs) {
    Write-Host "Stopping start.ps1 runner PID $($w.ProcessId)..." -ForegroundColor Cyan
    try { Stop-Process -Id $w.ProcessId -Force -ErrorAction Stop; Write-Host "  stopped." -ForegroundColor Green }
    catch { Write-Host "  failed: $($_.Exception.Message)" -ForegroundColor Red }
}

# Kill any python process running uvicorn app.main
$procs = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and $_.CommandLine -like '*uvicorn*app.main*' }

if (-not $procs) {
    Write-Host "No uvicorn processes found. Already stopped." -ForegroundColor Yellow
}

foreach ($p in $procs) {
    Write-Host "Stopping PID $($p.ProcessId)..." -ForegroundColor Cyan
    try {
        Stop-Process -Id $p.ProcessId -Force
        Write-Host "  killed." -ForegroundColor Green
    } catch {
        Write-Host "  failed: $_" -ForegroundColor Red
    }
}

# Stop any auto-start-worker background jobs left in the shell
Get-Job -Name "auto-start-worker" -ErrorAction SilentlyContinue |
    Remove-Job -Force -ErrorAction SilentlyContinue | Out-Null

if ($StopTunnel) {
    $cfs = Get-Process cloudflared -ErrorAction SilentlyContinue
    foreach ($c in $cfs) {
        Write-Host "Stopping cloudflared PID $($c.Id) (the next tunnel gets a new URL)..." -ForegroundColor Cyan
        try { Stop-Process -Id $c.Id -Force -ErrorAction Stop; Write-Host "  stopped." -ForegroundColor Green }
        catch { Write-Host "  failed: $($_.Exception.Message)" -ForegroundColor Red }
    }
} elseif (Get-Process cloudflared -ErrorAction SilentlyContinue) {
    Write-Host "Tunnel left running (same URL); -StopTunnel stops it." -ForegroundColor DarkGray
}

# Local PostgreSQL (DB_BACKEND=postgres only), after the server that uses it
if ((Get-EnvSetting 'DB_BACKEND' 'supabase').ToLower() -eq 'postgres' -and -not $KeepDb) {
    $pgCtl  = Join-Path (Get-EnvSetting 'PG_BIN' 'C:\GeoAI\pgsql\bin') 'pg_ctl.exe'
    $pgData = Get-EnvSetting 'PG_DATA' 'C:\GeoAI\pgdata'
    if (-not (Test-Path $pgCtl)) {
        Write-Host "pg_ctl.exe not found at $pgCtl - local PostgreSQL not stopped." -ForegroundColor Yellow
    } else {
        & $pgCtl status -D $pgData *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Local PostgreSQL is not running." -ForegroundColor Yellow
        } else {
            Write-Host "Stopping local PostgreSQL ($pgData)..." -ForegroundColor Cyan
            & $pgCtl stop -D $pgData -m fast -w -t 60 *> $null
            if ($LASTEXITCODE -eq 0) { Write-Host "  stopped." -ForegroundColor Green }
            else { Write-Host "  pg_ctl stop failed (exit $LASTEXITCODE) - see $pgData\log\postgres.log" -ForegroundColor Red }
        }
    }
}

Write-Host ""
Write-Host "Shutdown complete." -ForegroundColor Green
