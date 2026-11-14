# launch.ps1 - one-click start for the whole GeoAI demo (called by GeoAI_Start.bat).
#
#  1. installs / refreshes the GeoAI-Keepalive scheduled task (every 2 min + at logon)
#  2. starts the stack THROUGH that task, so nothing runs inside this window:
#     closing this window, a terminal or the Claude app cannot stop the services
#  3. waits for the model, makes sure the worker and the video pipeline are on
#  4. runs status.ps1 and checks every public page through the tunnel
#  5. prints the links and saves them to C:\GeoAI\tmp\links.txt
#
# Logs if anything goes wrong:
#   eval_results\launcher.log          this script
#   eval_results\keepalive.log         every automatic restart
#   eval_results\production_final.log  server output
#   C:\GeoAI\tmp\cloudflared.log       tunnel

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Log = Join-Path $Root 'eval_results\launcher.log'
$UrlFile = 'C:\GeoAI\tmp\tunnel_url.txt'
$LinksFile = 'C:\GeoAI\tmp\links.txt'
New-Item -ItemType Directory -Force 'C:\GeoAI\tmp' | Out-Null

function Say([string]$m, [string]$c = 'Gray') {
    Write-Host $m -ForegroundColor $c
    Add-Content -Path $Log -Value "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') $m" -Encoding utf8
}
function Get-Key {
    foreach ($l in Get-Content (Join-Path $Root '.env')) {
        if ($l -match '^\s*API_KEYS\s*=\s*(.*?)\s*$') {
            return (($Matches[1].Trim('"').Trim("'") -split ',') | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -First 1)
        }
    }
}
function Get-Health {
    try { return Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 8 } catch { return $null }
}

Say ''
Say '=== GeoAI launcher ===' 'Cyan'

# 1. Keep-alive task.
Remove-Item 'C:\GeoAI\tmp\keepalive.off' -ErrorAction SilentlyContinue
& (Join-Path $Root 'keepalive.ps1') -Install | Out-Null
if (Get-ScheduledTask -TaskName 'GeoAI-Keepalive' -ErrorAction SilentlyContinue) { Say '[OK]  keep-alive task installed (checks every 2 min)' 'Green' }
else { Say '[!!]  could not install the keep-alive task - services will still start, but will not self-heal' 'Red' }

# 2. Start through the task (outside this window).
if (Get-Health) { Say '[OK]  server already answering' 'Green' }
else {
    Say '[..]  starting services (model load takes about 1-2 minutes)...' 'Yellow'
    Start-ScheduledTask -TaskName 'GeoAI-Keepalive' -ErrorAction SilentlyContinue
}

# 3. Wait for the model.
$deadline = (Get-Date).AddMinutes(6)
$h = $null
while (-not $h -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    $h = Get-Health
    Write-Host '.' -NoNewline
}
Write-Host ''
if (-not $h) {
    Say '[!!]  server did not answer within 6 minutes. The keep-alive task keeps trying.' 'Red'
    Say '      Look at eval_results\production_final.log and eval_results\keepalive.log' 'Red'
} else {
    Say "[OK]  model loaded: $($h.model_name)" 'Green'
    $key = Get-Key
    $hdr = @{}; if ($key) { $hdr['X-API-Key'] = $key }
    # worker
    Start-Sleep -Seconds 10   # start.ps1's own auto-start fires shortly after the model loads
    try {
        $s = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/operator/status' -Headers $hdr -TimeoutSec 10
        if (-not $s.is_running) {
            Invoke-RestMethod -Uri 'http://127.0.0.1:8000/operator/start' -Method POST -Headers $hdr -TimeoutSec 30 | Out-Null
            Say '[OK]  worker started' 'Green'
        } else { Say '[OK]  worker running' 'Green' }
    } catch { Say "[!!]  worker check failed: $($_.Exception.Message)" 'Red' }
    # video pipeline
    try {
        $v = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/operator/video/status' -Headers $hdr -TimeoutSec 10
        if (-not $v.enabled) {
            Invoke-RestMethod -Uri 'http://127.0.0.1:8000/operator/video/enable' -Method POST -Headers $hdr -TimeoutSec 30 | Out-Null
            Say '[OK]  video pipeline enabled' 'Green'
        } else { Say '[OK]  video pipeline enabled' 'Green' }
    } catch { Say "[!!]  video pipeline check failed: $($_.Exception.Message)" 'Red' }
}

# 4. Tunnel: wait for its URL, then check every page through it.
$url = $null
$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline) {
    if ((Get-Process cloudflared -ErrorAction SilentlyContinue) -and (Test-Path $UrlFile)) {
        $url = (Get-Content $UrlFile -TotalCount 1).Trim()
        try { if ((Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 15).StatusCode -eq 200) { break } } catch {}
    }
    Start-Sleep -Seconds 5
}

Say ''
Say '--- automated checks (status.ps1) ---' 'Cyan'
& (Join-Path $Root 'status.ps1')

$pages = [ordered]@{
    'Map + condition reports' = '/gis/app/'
    'Expert review'           = '/expert/'
    'Try a photo'             = '/try/'
    'Operator (API key)'      = '/operator'
    'Dashboard (API key)'     = '/dashboard'
}
Say ''
Say '--- public links ---' 'Cyan'
$lines = @("GeoAI links, $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
$allOk = $true
if (-not $url) { Say '[!!]  no tunnel URL yet - the keep-alive task will start one; run this again in 2 minutes' 'Red'; $allOk = $false }
else {
    foreach ($name in $pages.Keys) {
        $u = "$url$($pages[$name])"
        $code = 0
        try { $code = (Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 20).StatusCode } catch { $code = 0 }
        $mark = if ($code -eq 200) { '[OK]' } else { $allOk = $false; '[!!]' }
        Say ("{0}  {1,-25} {2}" -f $mark, $name, $u) $(if ($code -eq 200) { 'Green' } else { 'Red' })
        $lines += ("{0,-25} {1}" -f $name, $u)
    }
    Set-Content -Path $LinksFile -Value $lines -Encoding utf8
    Say ''
    Say "Links saved to $LinksFile" 'Gray'
}
Say ''
if ($allOk) { Say 'ALL UP. You can close this window - the services keep running and restart themselves.' 'Green' }
else { Say 'Something is not up yet. Logs: eval_results\launcher.log, keepalive.log, production_final.log' 'Yellow' }
Say 'Operator key: C:\GeoAI\operator_api_key.txt   |   stop everything: .\stop.ps1 -StopTunnel' 'Gray'
