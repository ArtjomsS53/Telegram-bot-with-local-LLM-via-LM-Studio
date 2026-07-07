@echo off
REM ============================================================
REM  Start everything: bot + mini app server + tunnel.
REM  Each component opens in its own window.
REM
REM  BEFORE running: start LM Studio and enable its local
REM  server (http://localhost:1234). It does not auto-start.
REM ============================================================

echo ============================================
echo  Starting Telegram bot components
echo ============================================
echo.
echo  [!] Make sure LM Studio local server is running.
echo.

REM --- 1. Telegram bot ---
echo Starting bot...
start "Telegram Bot" /d "%~dp0" cmd /k "python telegram-local-llm-bot.py"

timeout /t 3 /nobreak >nul

REM --- 2. Mini app server (http://localhost:8080) ---
echo Starting mini app server...
start "Mini App Server" /d "%~dp0" cmd /k "python -m pip install -q -r miniapp\requirements-miniapp.txt & python -m uvicorn miniapp.server:app --host 0.0.0.0 --port 8080 --reload --reload-dir miniapp"

timeout /t 3 /nobreak >nul

REM --- 3. HTTPS tunnel (to open inside Telegram) ---
echo Starting Cloudflare tunnel...
start "Cloudflare Tunnel" /d "%~dp0miniapp" cmd /k "call tunnel_cloudflare.bat"

echo.
echo Done. Three windows opened:
echo   - Telegram Bot
echo   - Mini App Server  (test: http://localhost:8080)
echo   - Cloudflare Tunnel (gives https://...trycloudflare.com)
echo.
echo Paste the tunnel link into BotFather (see miniapp\README.md).
echo You can close this window.
echo.
pause
