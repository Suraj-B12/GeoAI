# ============================================================
# GeoAI — clean shutdown
# ============================================================
# Stops the uvicorn server gracefully (calls /operator/stop first to drain
# any in-flight worker, then kills the process).
#
# Usage:
#   .\stop.ps1            # graceful (drains worker first)
#   .\stop.ps1 -Force     # immediate kill
# ============================================================

param([switch]$Force, [int]$Port = 8000)

$ErrorActionPreference = "Continue"

if (-not $Force) {
    Write-Host "Asking worker to drain..."
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/operator/stop" `
            -Method POST -UseBasicParsing -TimeoutSec 30 | Out-Null
        Write-Host "  Drain request sent." -ForegroundColor Green
    } catch {
        Write-Host "  /operator/stop failed (server may already be down): $_" -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 2
}

# Kill any python process running uvicorn app.main
$procs = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and $_.CommandLine -like '*uvicorn*app.main*' }

if (-not $procs) {
    Write-Host "No uvicorn processes found. Already stopped." -ForegroundColor Yellow
    exit 0
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

Write-Host ""
Write-Host "Shutdown complete." -ForegroundColor Green
