# GeoAI / Pavement Distress Pipeline — startup script
# ----------------------------------------------------
# Usage (PowerShell):
#     .\start.ps1                  # server only (manual worker start via dashboard)
#     .\start.ps1 -AutoStartWorker # server + auto-start worker once model loads
#     .\start.ps1 -Tunnel          # also expose via cloudflared tunnel
#     .\start.ps1 -AutoStartWorker -Tunnel
#
# First run after cloning:
#     PowerShell may block scripts. Run once:
#         Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param(
    [switch]$AutoStartWorker,
    [switch]$Tunnel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# 1. Activate venv
$activate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "ERROR: venv not found at $activate" -ForegroundColor Red
    Write-Host "Recreate it with:  python -m venv venv  &&  .\venv\Scripts\activate  &&  python -m pip install -r requirements.txt"
    exit 1
}
& $activate

# 2. Sanity check .env
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env file missing in project root" -ForegroundColor Red
    Write-Host "Create it with SUPABASE_URL and SUPABASE_SERVICE_KEY"
    exit 1
}

# 3. Optional: launch cloudflared in a separate window for remote access
if ($Tunnel) {
    if (-not (Test-Path ".\cloudflared.exe")) {
        Write-Host "WARN: cloudflared.exe not found in project root — skipping tunnel" -ForegroundColor Yellow
    } else {
        Write-Host "Launching Cloudflare tunnel in a new window..." -ForegroundColor Cyan
        Start-Process powershell -ArgumentList "-NoExit","-Command",".\cloudflared.exe tunnel --url http://localhost:8000"
    }
}

# 4. Optional: auto-start worker once the API is up (background job)
if ($AutoStartWorker) {
    Start-Job -Name "auto-start-worker" -ScriptBlock {
        # Poll /health until ready, then start the worker
        $deadline = (Get-Date).AddMinutes(3)
        while ((Get-Date) -lt $deadline) {
            try {
                $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
                if ($r.StatusCode -eq 200) { break }
            } catch {}
            Start-Sleep -Seconds 2
        }
        try {
            Invoke-WebRequest -Uri "http://localhost:8000/operator/start" -Method POST -UseBasicParsing | Out-Null
            Write-Host "[auto-start] Worker started" -ForegroundColor Green
        } catch {
            Write-Host "[auto-start] Failed to start worker: $_" -ForegroundColor Red
        }
    } | Out-Null
    Write-Host "Worker will auto-start once /health responds (background job)" -ForegroundColor Cyan
}

# 5. Start the server (foreground, blocks until Ctrl+C)
Write-Host ""
Write-Host "Starting GeoAI server..." -ForegroundColor Green
Write-Host "  Dashboard: http://localhost:8000/operator"
Write-Host "  Health:    http://localhost:8000/health"
Write-Host "  Stop:      Ctrl+C"
Write-Host ""

python -m uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000 --workers 1

# When uvicorn exits (Ctrl+C), clean up background jobs
Get-Job -Name "auto-start-worker" -ErrorAction SilentlyContinue | Remove-Job -Force | Out-Null
