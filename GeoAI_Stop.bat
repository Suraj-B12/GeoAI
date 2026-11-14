@echo off
rem GeoAI one-click stop: pauses the keep-alive task, drains the worker, stops the
rem server, the Cloudflare tunnel and the local PostgreSQL. Start again with GeoAI_Start.bat.
title GeoAI stop
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" -StopTunnel
echo.
echo Stopping local PostgreSQL...
"C:\GeoAI\pgsql\bin\pg_ctl.exe" status -D "C:\GeoAI\pgdata" >nul 2>&1 && "C:\GeoAI\pgsql\bin\pg_ctl.exe" stop -D "C:\GeoAI\pgdata" -m fast
echo.
echo All GeoAI services stopped. They will not restart until you run GeoAI_Start.bat.
pause
