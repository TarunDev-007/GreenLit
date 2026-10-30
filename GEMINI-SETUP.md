# Gemini setup

GreenLit uses Gemini for every inference call. LangGraph, MCP and the local dashboard run on your computer; reference text, questions and answers are sent to Google for evaluation.

1. Run `setup.ps1` in your project folder (or double-click **Start GreenLit.cmd** for first setup).
2. Double-click **Set Gemini Key.cmd** and paste the key into its hidden terminal prompt. It writes the ignored `.env` file and checks an actual generation.
3. Restart **Start GreenLit.cmd** and **Start Math Chatbot.cmd** if they were running.
4. Open http://127.0.0.1:8765 and click **Check model**.

Terminal alternative:

```powershell
.\.venv\Scripts\python.exe -m greenlit.cli configure-gemini --model gemini-3.5-flash-lite
.\.venv\Scripts\python.exe -m greenlit.cli check-model
.\.venv\Scripts\python.exe -m greenlit.cli serve
```

Keep the API key in your local `.env`. Do not put it in the target API URL, dashboard reference fields, source code, or GitHub. The chatbot target's API URL is separate from the Gemini evaluator endpoint.

An empty key shows setup required. Invalid keys, missing model access, timeouts and exhausted quota stop inference with a visible error; the app does not substitute demo answers or another provider. Model listing alone does not confirm successful generation. Once a key is configured, a live probe is needed to verify access.

See [Google's API key guide](https://ai.google.dev/gemini-api/docs/api-key) and [pricing](https://ai.google.dev/gemini-api/docs/pricing).

The default is 6 questions, 1 trial and a 13-second shared interval between Gemini request starts. Set `GEMINI_REQUEST_INTERVAL_SECONDS` in `.env` according to your active Google project limits. Larger evaluations require more quota; creating another key in the same project does not provide separate limits.
