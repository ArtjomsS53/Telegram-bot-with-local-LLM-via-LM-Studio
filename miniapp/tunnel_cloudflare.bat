@echo off
REM ============================================================
REM  Free HTTPS tunnel to the local server via Cloudflare
REM  (cloudflared). No account required.
REM
REM  1) Download cloudflared.exe:
REM     https://github.com/cloudflare/cloudflared/releases/latest
REM     (file cloudflared-windows-amd64.exe -> rename to cloudflared.exe)
REM  2) Put cloudflared.exe next to this .bat (or add it to PATH).
REM  3) Start run_miniapp.bat first, then run this file.
REM
REM  cloudflared prints a link like https://xxxx.trycloudflare.com
REM  Paste it into BotFather as the Mini App URL.
REM ============================================================
cd /d "%~dp0"

where cloudflared >nul 2>nul
if %errorlevel%==0 (
  cloudflared tunnel --url http://localhost:8080
) else if exist cloudflared.exe (
  cloudflared.exe tunnel --url http://localhost:8080
) else (
  echo cloudflared not found.
  echo Download cloudflared.exe and put it next to this file:
  echo https://github.com/cloudflare/cloudflared/releases/latest
  pause
)
