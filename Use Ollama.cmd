@echo off
cd /d "%~dp0"
echo Ensure Ollama is running and llama3.2:3b is installed.
".venv\Scripts\python.exe" -m greenlit.cli configure-ollama --model llama3.2:3b
if errorlevel 1 goto done
".venv\Scripts\python.exe" -m greenlit.cli check-model
:done
echo Restart GreenLit and the math chatbot to use the selected model.
pause
