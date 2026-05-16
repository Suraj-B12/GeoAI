# ============================================================
# GeoAI / Pavement Distress Pipeline — failsafe startup script
# ============================================================
# Single command to launch the entire production stack.
#
# Usage:
#   .\start.ps1                           # server (manual pipeline Start from UI)
#   .\start.ps1 -AutoStart                # server + auto-start the worker pipeline
#   .\start.ps1 -Tunnel                   # also expose via Cloudflare tunnel
#   .\start.ps1 -Watchdog                 # auto-restart on crash (infinite loop)
#   .\start.ps1 -AutoStart -Tunnel -Watchdog  # full production runner
#
# First-time setup (run once after cloning):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# Stop with Ctrl+C — propagates through the watchdog and stops uvicorn.
# ============================================================

param(
    [switch]$AutoStart,         # auto-start the worker pipeline after model loads
    [switch]$Tunnel,            # spin up cloudflared in a separate window
    [switch]$Watchdog,          # auto-restart server on crash
    [switch]$SkipChecks,        # skip pre-flight (use only for fast iteration)
    [int]$Port = 8000,          # uvicorn port
    [string]$Host = '0.0.0.0'   # uvicorn host
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# ============================================================
# Helpers
# ============================================================
function Write-OK    { param($msg) Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-WARN  { param($msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-FAIL  { param($msg) Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Step  { param($msg) Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

function Test-PortFree {
    param([int]$Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.ReceiveTimeout = 500; $tcp.SendTimeout = 500
        $tcp.Connect('127.0.0.1', $Port)
        $tcp.Close()
        return $false  # something is listening
    } catch {
        return $true  # port is free
    }
}

function Stop-StalePython {
    # Kill any leftover python processes that might hold the port or GPU.
    # Safe — only kills processes whose command-line includes "uvicorn app.main".
    $procs = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*uvicorn*app.main*' }
    foreach ($p in $procs) {
        Write-WARN "killing stale uvicorn PID $($p.ProcessId)"
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
    }
    if ($procs) { Start-Sleep -Seconds 2 }
}

# ============================================================
# Pre-flight checks
# ============================================================
if (-not $SkipChecks) {
    Write-Step "Pre-flight checks"

    # 1. venv exists
    $venvPy = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        Write-FAIL "venv not found at $venvPy"
        Write-Host "    Recreate it:  python -m venv venv;  .\venv\Scripts\Activate.ps1;  uv pip install -r requirements.txt"
        exit 1
    }
    Write-OK "venv found"

    # 2. .env present + has the required keys
    if (-not (Test-Path ".env")) {
        Write-FAIL ".env file missing"
        Write-Host "    Copy .env.example -> .env and fill in SUPABASE_SERVICE_KEY."
        exit 1
    }
    $envText = Get-Content .env -Raw
    $required = @('SUPABASE_URL', 'SUPABASE_SERVICE_KEY')
    foreach ($k in $required) {
        if ($envText -notmatch "(?m)^\s*$k\s*=\s*\S") {
            Write-FAIL ".env missing required key: $k"
            exit 1
        }
    }
    Write-OK ".env present with required keys"

    # 3. Verify production switches set (DISABLE_ADAPTER + PROMPTS_VERSION)
    if ($envText -notmatch '(?m)^\s*DISABLE_ADAPTER\s*=\s*true') {
        Write-WARN "DISABLE_ADAPTER is not 'true' — adapter pathway will be used"
        Write-Host "         For production (Improved Baseline), set DISABLE_ADAPTER=true in .env"
    } else {
        Write-OK "DISABLE_ADAPTER=true (Improved Baseline production config)"
    }
    if ($envText -match '(?m)^\s*PROMPTS_VERSION\s*=\s*(v\d)') {
        Write-OK "PROMPTS_VERSION=$($Matches[1])"
    }

    # 4. Cloudinary keys (warn only — dashboard delete falls back gracefully)
    $hasCloudinary = ($envText -match 'CLOUDINARY_CLOUD_NAME=\S') -and
                     ($envText -match 'CLOUDINARY_API_KEY=\S') -and
                     ($envText -match 'CLOUDINARY_API_SECRET=\S')
    if ($hasCloudinary) { Write-OK "Cloudinary creds present (dashboard delete fully functional)" }
    else { Write-WARN "Cloudinary creds missing — dashboard delete will only remove Supabase row" }

    # 5. Port free (kill stale uvicorn if any)
    if (-not (Test-PortFree $Port)) {
        Write-WARN "Port $Port is in use — attempting to free it"
        Stop-StalePython
        if (-not (Test-PortFree $Port)) {
            Write-FAIL "Port $Port still in use after cleanup. Another process is holding it."
            Write-Host "    Find it:  Get-NetTCPConnection -LocalPort $Port"
            exit 1
        }
    }
    Write-OK "Port $Port is free"

    # 6. GPU sanity check (optional — pass if nvidia-smi works)
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $gpu = & nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-OK "GPU: $($gpu -join ' | ')"
        } else {
            Write-WARN "nvidia-smi failed — GPU may be unavailable"
        }
    } else {
        Write-WARN "nvidia-smi not found — running on CPU only?"
    }
}

# ============================================================
# Optional: Cloudflare tunnel (background)
# ============================================================
if ($Tunnel) {
    Write-Step "Cloudflare tunnel"
    $cf = Join-Path $ProjectRoot "cloudflared.exe"
    if (-not (Test-Path $cf)) {
        Write-WARN "cloudflared.exe not found at $cf — skipping tunnel"
        Write-Host "    Download:  Invoke-WebRequest -OutFile cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    } else {
        Write-OK "Launching cloudflared in a new window"
        Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$ProjectRoot'; .\cloudflared.exe tunnel --url http://localhost:$Port"
    }
}

# ============================================================
# Optional: auto-start worker (background job, fires after server is healthy)
# ============================================================
if ($AutoStart) {
    Write-Step "Worker auto-start scheduled"
    Start-Job -Name "auto-start-worker" -ArgumentList $Port -ScriptBlock {
        param($Port)
        $deadline = (Get-Date).AddMinutes(5)
        while ((Get-Date) -lt $deadline) {
            try {
                $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" `
                    -UseBasicParsing -TimeoutSec 3
                if ($r.StatusCode -eq 200) {
                    $h = $r.Content | ConvertFrom-Json
                    if ($h.model_loaded) { break }
                }
            } catch { }
            Start-Sleep -Seconds 3
        }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/operator/start" `
                -Method POST -UseBasicParsing
            Write-Host "[auto-start] worker started ($($r.StatusCode))" -ForegroundColor Green
        } catch {
            Write-Host "[auto-start] FAILED: $_" -ForegroundColor Red
        }
    } | Out-Null
    Write-OK "worker will auto-start once /health reports model_loaded=true"
}

# ============================================================
# Banner — URLs printed clearly
# ============================================================
Write-Step "Endpoints (open in browser)"
$localBase = "http://localhost:$Port"
Write-Host "  Operator UI:      $localBase/operator     " -NoNewline
Write-Host "(Start/Stop pipeline, live metrics)" -ForegroundColor DarkGray
Write-Host "  Image Dashboard:  $localBase/dashboard    " -NoNewline
Write-Host "(browse all uploads, delete, re-classify)" -ForegroundColor DarkGray
Write-Host "  Health:           $localBase/health       " -NoNewline
Write-Host "(JSON status of model + pipeline config)" -ForegroundColor DarkGray
Write-Host "  API docs:         $localBase/docs         " -NoNewline
Write-Host "(OpenAPI / Swagger interactive)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Stop:             Ctrl+C in this window"   -ForegroundColor DarkGray
if ($Watchdog) {
    Write-Host "  Watchdog:         ON — server will auto-restart on crash" -ForegroundColor Yellow
}
Write-Host ""

# ============================================================
# Run uvicorn (foreground)
# ============================================================
$venvPy = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$uvicornArgs = @(
    "-m", "uvicorn", "app.main:app",
    "--env-file", ".env",
    "--host", $Host,
    "--port", $Port,
    "--workers", "1"
)

if ($Watchdog) {
    # Failsafe loop — restart server on any non-Ctrl+C exit, with backoff.
    $crashCount = 0
    while ($true) {
        Write-Step "Starting uvicorn (attempt #$($crashCount + 1))"
        & $venvPy @uvicornArgs
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0 -or $exitCode -eq -1073741510) {
            # 0 = clean exit; -1073741510 = Ctrl+C on Windows
            Write-Host ""
            Write-Host "uvicorn exited cleanly. Stopping watchdog." -ForegroundColor Yellow
            break
        }
        $crashCount++
        $backoff = [Math]::Min(60, 5 * $crashCount)  # 5s, 10s, 15s, ..., cap at 60s
        Write-FAIL "uvicorn crashed (exit $exitCode). Restarting in $backoff seconds (crash #$crashCount)..."
        Stop-StalePython
        Start-Sleep -Seconds $backoff
    }
} else {
    & $venvPy @uvicornArgs
}

# Cleanup
Get-Job -Name "auto-start-worker" -ErrorAction SilentlyContinue | Remove-Job -Force | Out-Null
Write-Host ""
Write-Host "Shutdown complete." -ForegroundColor Yellow
