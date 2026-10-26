@echo off
cd /d "%~dp0"
echo Math chatbot API: http://127.0.0.1:9000/chat
echo Keep this window open. Ctrl+C stops the chatbot.
".venv\Scripts\python.exe" -m greenlit.math_chatbot
pause
