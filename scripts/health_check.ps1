# ============================================================
# GeoAI health check — single-shot diagnostic
# ============================================================
# Verifies the production stack is functional. Designed to be runnable
# while the server is up, OR from cron/scheduled task.
#
# Exit codes:
#   0 = all checks passed
#   1 = at least one check failed
#
# Usage:
#   .\scripts\health_check.ps1
#   .\scripts\health_check.ps1 -Port 8000
# ============================================================

param([int]$Port = 8000)

$ErrorActionPreference = "Continue"  # don't bail on first failure
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

$pass = 0; $fail = 0; $warn = 0
function Check-OK    { param($msg) Write-Host "  [PASS] $msg" -ForegroundColor Green; $script:pass++ }
function Check-FAIL  { param($msg) Write-Host "  [FAIL] $msg" -ForegroundColor Red;   $script:fail++ }
function Check-WARN  { param($msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow; $script:warn++ }

Write-Host "================================================================"
Write-Host " GeoAI health check  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
Write-Host "================================================================"
Write-Host ""

# 1. Server reachable + healthy
Write-Host "[1/6] Server reachability"
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 5
    $h = $r.Content | ConvertFrom-Json
    if ($h.status -eq "ok") {
        Check-OK "/health responded ok (HTTP $($r.StatusCode))"
    } else {
        Check-FAIL "/health returned status=$($h.status)"
    }
    if ($h.model_loaded) { Check-OK "model loaded: $($h.model_name)" }
    else { Check-FAIL "model NOT loaded" }
    if ($h.adapter_disabled_in_config) { Check-OK "adapter disabled (Improved Baseline production)" }
    elseif ($h.adapter_loaded) { Check-WARN "adapter is loaded — running experimental config" }
    if ($h.prompts_version) { Check-OK "prompts: $($h.prompts_version)" }
    if ($h.pavement_filter_enabled) { Check-OK "pavement pre-filter enabled" }
} catch {
    Check-FAIL "server not reachable on port $Port: $_"
}

# 2. UIs serve correctly
Write-Host ""
Write-Host "[2/6] UI endpoints"
foreach ($path in @('/operator', '/dashboard')) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port$path" -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200 -and $r.Content -like '*<!DOCTYPE html>*') {
            Check-OK "$path responds with HTML ($($r.RawContentLength) bytes)"
        } else {
            Check-FAIL "$path returned HTTP $($r.StatusCode)"
        }
    } catch {
        Check-FAIL "$path unreachable: $_"
    }
}

# 3. Dashboard API endpoints
Write-Host ""
Write-Host "[3/6] Dashboard API"
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/dashboard/summary" -UseBasicParsing -TimeoutSec 5
    $j = $r.Content | ConvertFrom-Json
    Check-OK "/dashboard/summary: $($j.total) total assessments"
    if ($j.by_status) {
        $statuses = ($j.by_status | Get-Member -MemberType NoteProperty | ForEach-Object {
            "$($_.Name)=$($j.by_status.$($_.Name))"
        }) -join ', '
        Check-OK "  status counts: $statuses"
    }
} catch {
    Check-FAIL "/dashboard/summary failed: $_"
}

# 4. Worker / operator state
Write-Host ""
Write-Host "[4/6] Worker state"
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/operator/status" -UseBasicParsing -TimeoutSec 5
    $s = $r.Content | ConvertFrom-Json
    if ($s.is_running) { Check-OK "worker is running (id $($s.worker_id))" }
    else { Check-WARN "worker is NOT running — start it from /operator UI" }
    if ($s.last_heartbeat_at) {
        $hbAge = ((Get-Date) - [datetime]::Parse($s.last_heartbeat_at)).TotalSeconds
        if ($hbAge -lt 60) { Check-OK "heartbeat fresh ($([int]$hbAge)s old)" }
        else { Check-WARN "heartbeat stale ($([int]$hbAge)s old)" }
    }
} catch {
    Check-FAIL "/operator/status failed: $_"
}

# 5. Supabase reachability (round-trip via the server)
Write-Host ""
Write-Host "[5/6] Supabase round-trip"
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/operator/metrics" -UseBasicParsing -TimeoutSec 10
    $m = $r.Content | ConvertFrom-Json
    if ($m.queue -and $m.queue.pending_count -ne $null) {
        Check-OK "Supabase reachable (queue: pending=$($m.queue.pending_count), processing=$($m.queue.processing_count))"
    } else {
        Check-WARN "/operator/metrics returned without queue data"
    }
} catch {
    Check-FAIL "/operator/metrics failed: $_"
}

# 6. GPU sanity
Write-Host ""
Write-Host "[6/6] GPU"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1
        Check-OK "nvidia-smi: $($gpu -join ' | ')"
    } catch {
        Check-WARN "nvidia-smi failed: $_"
    }
} else {
    Check-WARN "nvidia-smi not in PATH"
}

# Summary
Write-Host ""
Write-Host "================================================================"
Write-Host " Result: $pass pass, $warn warn, $fail fail"
Write-Host "================================================================"
if ($fail -gt 0) { exit 1 } else { exit 0 }
