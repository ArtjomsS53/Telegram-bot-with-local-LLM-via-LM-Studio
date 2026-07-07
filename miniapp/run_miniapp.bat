@echo off
REM ============================================================
REM  Run the Telegram Mini App backend (Windows).
REM ============================================================
cd /d "%~dp0.."

echo === Telegram Mini App backend ===
echo Installing dependencies...
python -m pip install -q -r miniapp\requirements-miniapp.txt

echo.
echo Server: http://localhost:8080
echo Open this address in a browser to test locally.
echo Press Ctrl+C to stop.
echo.

python -m uvicorn miniapp.server:app --host 0.0.0.0 --port 8080 --reload --reload-dir miniapp

pause
