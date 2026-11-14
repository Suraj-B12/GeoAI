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
#   .\start.ps1 -AutoStart -Tunnel -Watchdog -Detach
#                                         # the same, hidden in the background, logging to
#                                         # eval_results\production_final.log (use this after a reboot)
#   .\status.ps1                          # one-screen check: model, worker, video, GPU, tunnel, GPU lock
#
# -AutoStart sends the first key in API_KEYS (.env) as X-API-Key and fires after
# EVERY watchdog relaunch, not only the first; result lines go to eval_results\autostart.log.
# -Tunnel starts cloudflared hidden (unless one is already running) and writes the
# public URL to TUNNEL_URL_FILE (default C:\GeoAI\tmp\tunnel_url.txt); a quick tunnel's
# URL changes every time cloudflared starts.
#
# DB_BACKEND=postgres, or a DATABASE_URL on this machine (the video pipeline's
# store): the local PostgreSQL cluster (PG_BIN / PG_DATA, default
# C:\GeoAI\pgsql\bin and C:\GeoAI\pgdata) is started first if it is not
# running, and before every watchdog restart. With DB_BACKEND=supabase a start
# failure is only a warning. See SETUP.md "Local PostgreSQL".
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
    [switch]$Detach,            # relaunch this script hidden in the background, output to $LogFile
    [string]$LogFile = '',      # with -Detach; default eval_results\production_final.log
    [int]$Port = 8000,          # uvicorn port
    [string]$BindHost = '0.0.0.0'   # uvicorn host. Not $Host: that is a read-only PowerShell automatic variable, and declaring it aborts the script before it runs.
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Remove-Item 'C:\GeoAI\tmp\keepalive.off' -ErrorAction SilentlyContinue   # resume keepalive.ps1
if ($Detach) {
    # Relaunch hidden with the same switches (minus -Detach) and all output in one log,
    # so the server outlives the terminal or agent session that started it.
    $running = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*uvicorn*app.main*' }
    if ($running) {
        Write-Host "A server is already running (PID $($running[0].ProcessId)). Not starting a second one." -ForegroundColor Yellow
        Write-Host "Check it with .\status.ps1; restart recipe: CLAUDE.md 'Restarting production'." -ForegroundColor Yellow
        exit 1
    }
    if (-not $LogFile) { $LogFile = Join-Path $ProjectRoot 'eval_results\production_final.log' }
    $argList = @()
    if ($AutoStart)  { $argList += '-AutoStart' }
    if ($Tunnel)     { $argList += '-Tunnel' }
    if ($Watchdog)   { $argList += '-Watchdog' }
    if ($SkipChecks) { $argList += '-SkipChecks' }
    $argList += @('-Port', $Port, '-BindHost', $BindHost)
    $cmd = "Set-Location '$ProjectRoot'; & '.\start.ps1' $($argList -join ' ') *> '$LogFile'"
    Start-Process -FilePath powershell.exe -WindowStyle Hidden `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd) | Out-Null
    Write-Host "Started in the background; log: $LogFile" -ForegroundColor Green
    Write-Host "The model takes about a minute to load. Then run .\status.ps1" -ForegroundColor Green
    exit 0
}

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

function Get-EnvSetting {
    # A setting from the process environment, else from .env (uvicorn's
    # --env-file does not override variables already set, so the same order).
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

function Start-LocalPostgres {
    # Start the local PostgreSQL cluster unless it is already running.
    # Returns $true once it accepts connections. Never prints DATABASE_URL.
    $ErrorActionPreference = 'Continue'   # pg_ctl writes progress to stderr
    $pgBin  = Get-EnvSetting 'PG_BIN'  'C:\GeoAI\pgsql\bin'
    $pgData = Get-EnvSetting 'PG_DATA' 'C:\GeoAI\pgdata'
    $pgCtl  = Join-Path $pgBin 'pg_ctl.exe'
    if (-not (Test-Path $pgCtl)) { Write-FAIL "pg_ctl.exe not found at $pgCtl (set PG_BIN)"; return $false }
    if (-not (Test-Path (Join-Path $pgData 'PG_VERSION'))) {
        Write-FAIL "no PostgreSQL data directory at $pgData (set PG_DATA)"; return $false
    }
    & $pgCtl status -D $pgData *> $null
    if ($LASTEXITCODE -eq 0) { Write-OK "PostgreSQL running ($pgData)"; return $true }
    $logDir = Join-Path $pgData 'log'
    New-Item -ItemType Directory -Force $logDir | Out-Null
    # Start-Process, not "& pg_ctl start": the postgres server inherits pg_ctl's
    # stdout/stderr, and when those are a PowerShell pipe (output redirected to a
    # log, as in production) PowerShell waits for them to close - forever.
    $logFile = Join-Path $logDir 'postgres.log'
    Start-Process -FilePath $pgCtl -WindowStyle Hidden `
        -ArgumentList @('start', '-D', "`"$pgData`"", '-l', "`"$logFile`"") | Out-Null
    # After an unclean stop PostgreSQL first fsyncs the data directory (35 s measured), then recovers.
    $deadline = (Get-Date).AddSeconds(180)
    do {
        Start-Sleep -Seconds 1
        & $pgCtl status -D $pgData *> $null
        $running = ($LASTEXITCODE -eq 0)
    } until ($running -or (Get-Date) -gt $deadline)
    $isReady = Join-Path $pgBin 'pg_isready.exe'
    while ($running -and (Test-Path $isReady) -and (Get-Date) -lt $deadline) {
        & $isReady -q -h localhost -p (Get-EnvSetting 'PG_PORT' '5432') *> $null
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 1
    }
    if (-not $running) { Write-FAIL "PostgreSQL did not start - see $logFile"; return $false }
    Write-OK "PostgreSQL started ($pgData, log: $logDir\postgres.log)"
    return $true
}

$DbBackend = (Get-EnvSetting 'DB_BACKEND' 'supabase').ToLower()
# The video pipeline stores its results in the local PostgreSQL (DATABASE_URL) whatever DB_BACKEND
# says, so a DATABASE_URL on this machine also starts it. With DB_BACKEND=supabase a failure is only
# a warning: citizen uploads don't need it, and the video toggle reports its store as unavailable.
$LocalPg = ($DbBackend -eq 'postgres') -or ((Get-EnvSetting 'DATABASE_URL' '') -match '@(localhost|127\.0\.0\.1|\[::1\])[:/]')

# ============================================================
# Pre-flight checks
# ============================================================
if (-not $SkipChecks) {
    Write-Step "Pre-flight checks"

    # 1. venv exists
    # From here on the script only launches a native process and reads its exit
# code. uvicorn writes its INFO log lines to stderr. With
# ErrorActionPreference=Stop, PowerShell 5.1 turns every stderr line into a
# terminating NativeCommandError as soon as output is redirected - to a log
# file, or by a service wrapper such as NSSM - so the server was killed on its
# very first log line. Crash detection below uses $LASTEXITCODE, not
# PowerShell errors, so Continue is the correct setting for this section.
$ErrorActionPreference = "Continue"
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
    if ($DbBackend -eq 'postgres') {
        $required = @('DATABASE_URL')
        Write-OK "DB_BACKEND=postgres (local PostgreSQL + local image store)"
    } elseif ($DbBackend -eq 'supabase') {
        $required = @('SUPABASE_URL', 'SUPABASE_SERVICE_KEY')
    } else {
        Write-FAIL "DB_BACKEND='$DbBackend' is not 'supabase' or 'postgres'"
        exit 1
    }
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
# Local PostgreSQL (DB_BACKEND=postgres, or a local DATABASE_URL for the video pipeline)
# ============================================================
if ($LocalPg) {
    Write-Step "Local PostgreSQL"
    if (-not (Start-LocalPostgres)) {
        if ($DbBackend -eq 'postgres') { exit 1 }
        Write-WARN "local PostgreSQL not running: the video pipeline cannot store results until it is"
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
    } elseif (Get-Process cloudflared -ErrorAction SilentlyContinue) {
        Write-OK "cloudflared already running - keeping it (URL in $(Get-EnvSetting 'TUNNEL_URL_FILE' 'C:\GeoAI\tmp\tunnel_url.txt'))"
    } else {
        $urlFile = Get-EnvSetting 'TUNNEL_URL_FILE' 'C:\GeoAI\tmp\tunnel_url.txt'
        $cfLog = Join-Path (Split-Path $urlFile) 'cloudflared.log'
        New-Item -ItemType Directory -Force (Split-Path $urlFile) | Out-Null
        Remove-Item $cfLog -ErrorAction SilentlyContinue
        Start-Process -FilePath $cf -WindowStyle Hidden `
            -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://localhost:$Port", '--logfile', $cfLog) | Out-Null
        $url = $null
        $deadline = (Get-Date).AddSeconds(60)
        while (-not $url -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 2
            if (Test-Path $cfLog) {
                $m = Select-String -Path $cfLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
                if ($m) { $url = $m.Matches[0].Value }
            }
        }
        if ($url) {
            Set-Content -Path $urlFile -Value $url -Encoding ascii
            Write-OK "Tunnel: $url  (saved to $urlFile)"
        } else {
            Write-WARN "cloudflared started but no URL within 60 s - see $cfLog"
        }
    }
}

# ============================================================
# Optional: auto-start worker (background job, fires after server is healthy)
# ============================================================
function Start-AutoStartJob {
    # Starts the worker once /health reports model_loaded. Called before every uvicorn launch
    # (also watchdog relaunches), so a crash no longer leaves the pipeline stopped.
    # Sends the first API_KEYS entry as X-API-Key (the operator endpoints need it once
    # API_KEYS is set); the key is passed to the job, never printed or logged.
    $key = ((Get-EnvSetting 'API_KEYS' '') -split ',' | ForEach-Object { $_.Trim() } |
            Where-Object { $_ } | Select-Object -First 1)
    $logPath = Join-Path $ProjectRoot 'eval_results\autostart.log'
    Get-Job -Name "auto-start-worker" -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
    Start-Job -Name "auto-start-worker" -ArgumentList $Port, $key, $logPath -ScriptBlock {
        param($Port, $Key, $LogPath)
        function Log($m) { Add-Content -Path $LogPath -Value "$(Get-Date -Format s) [auto-start] $m" }
        $loaded = $false
        $deadline = (Get-Date).AddMinutes(10)
        while ((Get-Date) -lt $deadline) {
            try {
                $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
                if ($r.StatusCode -eq 200 -and ($r.Content | ConvertFrom-Json).model_loaded) { $loaded = $true; break }
            } catch { }
            Start-Sleep -Seconds 3
        }
        if (-not $loaded) { Log "model not loaded within 10 min - worker NOT started"; return }
        $headers = @{}
        if ($Key) { $headers['X-API-Key'] = $Key }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/operator/start" -Method POST `
                -Headers $headers -UseBasicParsing -TimeoutSec 30
            Log "worker started (HTTP $($r.StatusCode))"
        } catch {
            Log "FAILED to start worker: $($_.Exception.Message)"
        }
    } | Out-Null
}

if ($AutoStart) {
    Write-Step "Worker auto-start scheduled"
    Write-OK "worker will auto-start once /health reports model_loaded=true (log: eval_results\autostart.log)"
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
# From here on the script only launches a native process and reads its exit
# code. uvicorn writes its INFO log lines to stderr. With
# ErrorActionPreference=Stop, PowerShell 5.1 turns every stderr line into a
# terminating NativeCommandError as soon as output is redirected - to a log
# file, or by a service wrapper such as NSSM - so the server was killed on its
# very first log line. Crash detection below uses $LASTEXITCODE, not
# PowerShell errors, so Continue is the correct setting for this section.
$ErrorActionPreference = "Continue"
$venvPy = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$uvicornArgs = @(
    "-m", "uvicorn", "app.main:app",
    "--env-file", ".env",
    "--host", $BindHost,
    "--port", $Port,
    "--workers", "1"
)

if ($Watchdog) {
    # Failsafe loop — restart server on any non-Ctrl+C exit, with backoff.
    $crashCount = 0
    while ($true) {
        Write-Step "Starting uvicorn (attempt #$($crashCount + 1))"
        if ($LocalPg) { [void](Start-LocalPostgres) }
        if ($AutoStart) { Start-AutoStartJob }
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
    if ($AutoStart) { Start-AutoStartJob }
    & $venvPy @uvicornArgs
}

# Cleanup
Get-Job -Name "auto-start-worker" -ErrorAction SilentlyContinue | Remove-Job -Force | Out-Null
Write-Host ""
Write-Host "Shutdown complete." -ForegroundColor Yellow
