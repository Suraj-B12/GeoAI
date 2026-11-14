@echo off
rem GeoAI one-click start: services, checks, self-restart, public links.
rem Double-click this file. You can close the window afterwards; the services keep running.
title GeoAI launcher
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
echo.
pause
