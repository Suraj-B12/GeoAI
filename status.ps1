# ============================================================
# status.ps1 - one-screen check of the production / demo stack
# ============================================================
# Read-only. Shows: model + P4 settings (vs the values CLAUDE.md requires), worker,
# video pipeline, local PostgreSQL, GPU memory, the GPU lock (C:\GeoAI\gpu.lock),
# the Cloudflare tunnel (URL + whether it answers) and the last auto-start line.
# Uses the first API_KEYS entry from .env for the operator endpoints; never prints it.
#
#   .\status.ps1            # port 8000
#   .\status.ps1 -Port 8000
# Exit code 0 = everything expected is up; 1 = something needs attention.
# ============================================================
param([int]$Port = 8000)

$ErrorActionPreference = 'Continue'
$ProjectRoot = $PSScriptRoot
$problems = 0
function OK($m)   { Write-Host "  [OK]   $m" -ForegroundColor Green }
function BAD($m)  { Write-Host "  [!!]   $m" -ForegroundColor Red; $script:problems++ }
function INFO($m) { Write-Host "  [..]   $m" -ForegroundColor DarkGray }

function Get-EnvSetting {
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
$key = ((Get-EnvSetting 'API_KEYS' '') -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -First 1)
$headers = @{}; if ($key) { $headers['X-API-Key'] = $key }
function Get-Json($path, [switch]$Auth) {
    try {
        $h = @{}; if ($Auth) { $h = $headers }
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port$path" -Headers $h -UseBasicParsing -TimeoutSec 8
        return ($r.Content | ConvertFrom-Json)
    } catch { return $null }
}

Write-Host ""
Write-Host "Server (http://localhost:$Port)" -ForegroundColor Cyan
$h = Get-Json '/health'
if (-not $h) {
    BAD "no answer on /health - start it: .\start.ps1 -AutoStart -Tunnel -Watchdog -Detach"
} elseif (-not $h.model_loaded) {
    BAD "server up, model still loading (about a minute after start)"
} else {
    OK "model loaded: $($h.model_name) on $($h.device), VRAM used $($h.vram_used_gb) GB"
    # The P4 production values (CLAUDE.md START HERE).
    $expect = [ordered]@{
        adapter_loaded = $false; quantization_bits = 4; max_image_pixels = 1048576; prompts_version = 'v2';
        taxonomy = 'IRC:82-2015'; stage2_confidence_mode = 'field'; stage2_mode = 'shadow';
        stage2_mode_effective = 'shadow'; stage1_safety_net = $true
    }
    $bad = @()
    foreach ($k in $expect.Keys) { if ("$($h.$k)" -ne "$($expect[$k])") { $bad += "$k=$($h.$k) (want $($expect[$k]))" } }
    if ($h.stage2_probe -and $h.stage2_probe.parity -and -not $h.stage2_probe.parity.ok) { $bad += 'stage2_probe.parity.ok=false' }
    if ($bad) { BAD ("not the P4 configuration: " + ($bad -join '; ')) } else { OK "P4 configuration (4-bit, 1024^2, v2, field gate, shadow probe, safety net, parity ok)" }
}

if ($h) {
    $s = Get-Json '/operator/status' -Auth
    if (-not $s) { BAD "/operator/status refused or failed (API key in .env API_KEYS?)" }
    elseif ($s.is_running) {
        $what = if ($s.current_image_id) { "busy: $($s.current_stage) on $($s.current_image_id)" } else { 'idle' }
        OK "worker running ($what), last processed $($s.last_processed_at)"
    } else { BAD "worker STOPPED (hold: $($s.hold_reason)) - POST /operator/start with the API key, or the Operator page" }

    $v = Get-Json '/operator/video/status' -Auth
    if (-not $v) { INFO "video pipeline status unavailable" }
    elseif ($v.enabled) { OK "video pipeline enabled (running=$($v.running), busy=$($v.busy)); counts: $(($v.counts | ConvertTo-Json -Compress))" }
    else { BAD "video pipeline disabled - POST /operator/video/enable with the API key" }
}

Write-Host ""
Write-Host "Machine" -ForegroundColor Cyan
$pgReady = Join-Path (Get-EnvSetting 'PG_BIN' 'C:\GeoAI\pgsql\bin') 'pg_isready.exe'
if (Test-Path $pgReady) {
    & $pgReady -q -h localhost -p (Get-EnvSetting 'PG_PORT' '5432') *> $null
    if ($LASTEXITCODE -eq 0) { OK "local PostgreSQL accepting connections" } else { BAD "local PostgreSQL not answering (start.ps1 starts it)" }
}
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $g = (& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits) -split ','
    if ($g.Count -ge 3) {
        $used = [int]$g[0]; $total = [int]$g[1]
        $msg = "GPU memory $used / $total MiB, utilisation $($g[2].Trim())%"
        if ($used -gt 0.92 * $total) { BAD "$msg - near full: Windows pages to RAM and everything slows 10-30x" } else { OK $msg }
    }
}
$lock = 'C:\GeoAI\gpu.lock'
if (Test-Path $lock) {
    $owner = (Get-Content (Join-Path $lock 'owner.txt') -ErrorAction SilentlyContinue) -join ' '
    $lockPid = if ($owner -match '(?i)pid\D{0,3}(\d+)') { [int]$Matches[1] } else { 0 }
    if ($lockPid -and -not (Get-Process -Id $lockPid -ErrorAction SilentlyContinue)) {
        BAD "GPU lock is STALE: owner PID $lockPid is gone ($owner). Check nvidia-smi, then: Remove-Item -Recurse $lock"
    } elseif ($lockPid) {
        INFO "GPU lock held by running PID ${lockPid}: $owner"
    } else {
        INFO "GPU lock HELD (no PID recorded): $owner  (remove only if that job is gone and nvidia-smi shows no other process)"
    }
} else { OK "GPU lock free" }

Write-Host ""
Write-Host "Public link" -ForegroundColor Cyan
$urlFile = Get-EnvSetting 'TUNNEL_URL_FILE' 'C:\GeoAI\tmp\tunnel_url.txt'
$cfRunning = [bool](Get-Process cloudflared -ErrorAction SilentlyContinue)
$url = if (Test-Path $urlFile) { (Get-Content $urlFile | Select-Object -First 1).Trim() } else { '' }
if (-not $cfRunning) { BAD "cloudflared not running - restart with start.ps1 -Tunnel (the URL will change)" }
elseif (-not $url) { BAD "cloudflared running but no URL in $urlFile" }
else {
    try {
        $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 15
        OK "$url  (answers $($r.StatusCode); pages: /gis/app/  /try/  /expert/  /operator)"
    } catch { BAD "$url does not answer: $($_.Exception.Message)" }
}

$auto = Join-Path $ProjectRoot 'eval_results\autostart.log'
if (Test-Path $auto) { INFO ("last auto-start: " + (Get-Content $auto -Tail 1)) }

Write-Host ""
if ($problems) { Write-Host "$problems item(s) need attention." -ForegroundColor Red; exit 1 }
Write-Host "All up." -ForegroundColor Green
exit 0
