@echo off
cd /d "%~dp0"
if not exist ".venv\greenlit-ready" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
echo GreenLit dashboard: http://127.0.0.1:8765
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m greenlit.cli serve
pause
