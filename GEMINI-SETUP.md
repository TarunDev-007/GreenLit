# Gemini setup

See README.md for the current implementation. The native Gemini REST API is used. The working model verified during this session is gemini-3.5-flash-lite.

```powershell
.\.venv\Scripts\python.exe -m greenlit.cli configure-gemini --model gemini-3.5-flash-lite
.\.venv\Scripts\python.exe -m greenlit.cli check-model
```

Enter keys only in the hidden terminal prompt. Configure-gemini saves the local ignored .env and reloads the dashboard. Use Check Gemini in the dashboard to test actual generation. Model listing alone does not establish that generation works.

Production demo mode is disabled. Test fixtures are for the automated test suite only. Real evaluations require the user's application API, purpose, and supporting reference/evidence where relevant.
