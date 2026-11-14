# keepalive.ps1 - brings the production stack back if any part of it is down.
#
# Run every 2 minutes by the Windows scheduled task "GeoAI-Keepalive" (and at logon),
# so it runs OUTSIDE any terminal or agent session: closing a terminal / the Claude app
# can no longer take the server, PostgreSQL and the tunnel down with it.
#
#   server (uvicorn) missing  -> .\start.ps1 -AutoStart -Tunnel -Watchdog -Detach
#   PostgreSQL down           -> pg_ctl start
#   cloudflared missing, or the public URL fails 3 checks in a row -> new tunnel (URL changes,
#                                written to C:\GeoAI\tmp\tunnel_url.txt)
#
# Paused while C:\GeoAI\tmp\keepalive.off exists (stop.ps1 creates it, start.ps1 removes it).
# Install / remove the task:  .\keepalive.ps1 -Install   /   .\keepalive.ps1 -Uninstall
# Log: eval_results\keepalive.log

param([switch]$Install, [switch]$Uninstall)

$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName = 'GeoAI-Keepalive'
$TmpDir = 'C:\GeoAI\tmp'
$OffFlag = Join-Path $TmpDir 'keepalive.off'
$FailFile = Join-Path $TmpDir 'keepalive_tunnel_fails.txt'
$Log = Join-Path $ProjectRoot 'eval_results\keepalive.log'
New-Item -ItemType Directory -Force $TmpDir | Out-Null

if ($Install) {
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ProjectRoot\keepalive.ps1`"" `
        -WorkingDirectory $ProjectRoot
    $t1 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 2) -RepetitionDuration (New-TimeSpan -Days 3650)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($t1, $t2) -Settings $settings `
        -Description 'Keeps the GeoAI server, PostgreSQL and Cloudflare tunnel running' -Force | Out-Null
    Write-Host "Installed scheduled task $TaskName (every 2 min + at logon)."
    exit 0
}
if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task $TaskName."
    exit 0
}

function Log([string]$m) {
    $line = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') $m"
    Add-Content -Path $Log -Value $line -Encoding utf8
}

if (Test-Path $OffFlag) { exit 0 }

# One run at a time.
$mutex = New-Object System.Threading.Mutex($false, 'Global\GeoAI-Keepalive')
if (-not $mutex.WaitOne(0)) { exit 0 }
try {
    function Get-EnvSetting([string]$Name, [string]$Default = '') {
        $f = Join-Path $ProjectRoot '.env'
        if (Test-Path $f) {
            foreach ($l in Get-Content $f) {
                if ($l -match "^\s*$Name\s*=\s*(.*?)\s*$") { return $Matches[1].Trim('"').Trim("'") }
            }
        }
        return $Default
    }

    # 1. Server. A uvicorn process that is still loading the model counts as up.
    $uv = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn*app.main*' }
    $wd = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*start.ps1*-Watchdog*' }
    if (-not $uv -and -not $wd) {
        Log 'server and watchdog not running -> start.ps1 -AutoStart -Tunnel -Watchdog -Detach'
        Set-Location $ProjectRoot
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot 'start.ps1') `
            -AutoStart -Tunnel -Watchdog -Detach 2>&1 | ForEach-Object { Log "  $_" }
        exit 0   # start.ps1 brings PostgreSQL and the tunnel too; check again next run
    }

    # 1b. Hung server: the process exists but /health has not answered for 3 checks (~6 min,
    # far longer than the ~70 s model load). Kill the python process; the watchdog relaunches it.
    $HangFile = Join-Path $TmpDir 'keepalive_server_fails.txt'
    if ($uv) {
        $up = $false
        try { $up = ((Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 20).StatusCode -eq 200) } catch {}
        if ($up) { Remove-Item $HangFile -ErrorAction SilentlyContinue }
        else {
            $n = 1; if (Test-Path $HangFile) { $n = [int](Get-Content $HangFile) + 1 }
            Set-Content $HangFile $n
            Log "server process present but /health not answering ($n in a row)"
            if ($n -ge 3) {
                Remove-Item $HangFile -ErrorAction SilentlyContinue
                if ($wd) {
                    Log 'killing the hung uvicorn; the watchdog will relaunch it'
                    $uv | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                } else {
                    Log 'hung uvicorn without a watchdog -> kill and start.ps1 -Detach'
                    $uv | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                    Start-Sleep -Seconds 5
                    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot 'start.ps1') `
                        -AutoStart -Tunnel -Watchdog -Detach 2>&1 | ForEach-Object { Log "  $_" }
                }
                exit 0
            }
        }
    }

    # 2. PostgreSQL.
    $pgBin = Get-EnvSetting 'PG_BIN' 'C:\GeoAI\pgsql\bin'
    $pgData = Get-EnvSetting 'PG_DATA' 'C:\GeoAI\pgdata'
    $pgCtl = Join-Path $pgBin 'pg_ctl.exe'
    if (Test-Path $pgCtl) {
        & $pgCtl status -D $pgData *> $null
        if ($LASTEXITCODE -ne 0) {
            Log 'PostgreSQL down -> pg_ctl start'
            Start-Process -FilePath $pgCtl -WindowStyle Hidden `
                -ArgumentList @('start', '-D', "`"$pgData`"", '-l', "`"$(Join-Path $pgData 'log\postgres.log')`"") | Out-Null
        }
    }

    # 3. Tunnel.
    $urlFile = Get-EnvSetting 'TUNNEL_URL_FILE' (Join-Path $TmpDir 'tunnel_url.txt')
    $cf = Join-Path $ProjectRoot 'cloudflared.exe'
    $needTunnel = $false
    if (-not (Get-Process cloudflared -ErrorAction SilentlyContinue)) {
        Log 'cloudflared not running'
        $needTunnel = $true
    } elseif ((Test-Path $urlFile) -and $uv) {
        $url = (Get-Content $urlFile -TotalCount 1).Trim()
        $ok = $false
        try { $ok = ((Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 20).StatusCode -eq 200) } catch {}
        $local = $false
        try { $local = ((Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 10).StatusCode -eq 200) } catch {}
        if ($ok -or -not $local) {
            Remove-Item $FailFile -ErrorAction SilentlyContinue   # fine, or server still loading
        } else {
            $n = 1; if (Test-Path $FailFile) { $n = [int](Get-Content $FailFile) + 1 }
            Set-Content $FailFile $n
            Log "public URL failed ($n in a row) while localhost answers"
            if ($n -ge 3) {
                Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
                Remove-Item $FailFile -ErrorAction SilentlyContinue
                $needTunnel = $true
            }
        }
    }
    if ($needTunnel -and (Test-Path $cf)) {
        $cfLog = Join-Path (Split-Path $urlFile) 'cloudflared.log'
        Remove-Item $cfLog -ErrorAction SilentlyContinue
        Start-Process -FilePath $cf -WindowStyle Hidden `
            -ArgumentList @('tunnel', '--no-autoupdate', '--url', 'http://localhost:8000', '--logfile', $cfLog) | Out-Null
        $new = $null; $deadline = (Get-Date).AddSeconds(60)
        while (-not $new -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 2
            if (Test-Path $cfLog) {
                $m = Select-String -Path $cfLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
                if ($m) { $new = $m.Matches[0].Value }
            }
        }
        if ($new) {
            Set-Content -Path $urlFile -Value $new -Encoding ascii; Log "new tunnel: $new"
            $paths = [ordered]@{ 'Map + condition reports' = '/gis/app/'; 'Expert review' = '/expert/'; 'Try a photo' = '/try/'; 'Operator (API key)' = '/operator'; 'Dashboard (API key)' = '/dashboard' }
            $lines = @("GeoAI links, $(Get-Date -Format 'yyyy-MM-dd HH:mm') (tunnel restarted by keep-alive)")
            foreach ($k in $paths.Keys) { $lines += ("{0,-25} {1}{2}" -f $k, $new, $paths[$k]) }
            Set-Content -Path (Join-Path $TmpDir 'links.txt') -Value $lines -Encoding utf8
        }
        else { Log 'cloudflared started but no URL within 60 s' }
    }
} finally {
    $mutex.ReleaseMutex()
}
