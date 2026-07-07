@echo off
REM ============================================================
REM  Restart ONLY the Mini App server (port 8080).
REM  The Cloudflare Tunnel keeps running -> the public URL
REM  does NOT change, so no need to edit BotFather again.
REM
REM  Just run this file. It frees port 8080 and starts fresh.
REM  Keep the "Cloudflare Tunnel" window open!
REM ============================================================
cd /d "%~dp0"

echo Freeing port 8080 (stopping old server if running)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>nul

echo Starting Mini App server (auto-reload on code changes)...
echo Keep the Cloudflare Tunnel window open - the URL stays the same.
echo.
python -m uvicorn miniapp.server:app --host 0.0.0.0 --port 8080 --reload --reload-dir miniapp

pause
