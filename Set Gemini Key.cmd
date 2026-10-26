@echo off
cd /d "%~dp0"
if not exist ".venv\greenlit-ready" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m greenlit.cli configure-gemini
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m greenlit.cli check-model
pause
