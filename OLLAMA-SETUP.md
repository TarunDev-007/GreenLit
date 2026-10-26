# GreenLit with local Ollama

The selected evaluation provider is Ollama, using the installed `llama3.2:3b` model. An API key is not needed. Gemini and OpenAI are not called in this mode.

The workflow is: describe the application and provide its API -> Llama plans tests -> Llama generates new questions -> MCP sends questions to the supplied API -> Llama judges actual answers -> Python measures operations and calculates GO/REVIEW/NO-GO. LangGraph still coordinates the workflow.

## Start from PowerShell

If Ollama is already running, keep it running. Otherwise, in a terminal:

```powershell
ollama serve
```

In another terminal:

```powershell
# Open a terminal in your cloned GreenLit folder after running setup.ps1:
.\.venv\Scripts\python.exe -m greenlit.cli check-model
.\.venv\Scripts\python.exe -m greenlit.cli serve
```

Open http://127.0.0.1:8765 and click **Check model**. To select the model again, run `Use Ollama.cmd`, or:

```powershell
.\.venv\Scripts\python.exe -m greenlit.cli configure-ollama --model llama3.2:3b
```

Restart GreenLit and the math chatbot after changing providers.

## Test the included real math chatbot

In a separate terminal in the same project folder:

```powershell
.\.venv\Scripts\python.exe -m greenlit.math_chatbot
```

- Application type: LLM / chatbot.
- API URL: `http://127.0.0.1:9000/chat`.
- Answer field: `answer`.
- Request template: `{"message":"{{question}}"}`.
- Multi-turn template: `{"messages":"{{messages}}"}`.
- Reference: upload `greenlit/examples/math-reference.md`.
- Timeout: 120 seconds. Start with 6 questions, one trial and load testing disabled.
- API credential variable: leave empty.

This target generates answers using the same local Llama model. Other real LLM, RAG or agent APIs can also be supplied as targets, with their matching request/response configuration.

## Practical limits

Question generation uses six focused categories, one short question at a time with schema-constrained output. Judgments run one at a time. Local generation can take several minutes; avoid starting simultaneous model checks or other local inference while an evaluation runs. The model stays loaded for ten minutes after a request.

The 16,384-token context uses a conservative input budget. Long references/history are rejected with a clear error rather than silently truncated. Shorten the document if prompted.

Llama 3.2 3B can misunderstand scenarios, generate weak expected answers, or judge incorrectly. Its confidence scores are not calibrated. The included math target and judge share a model, so their errors can correlate. Review results and validate important cases against independent labels. A local model does not guarantee full coverage or a GO verdict.

Missing retrieval evidence, tool traces, pricing, repeated trials or optional load measurements still appear as unmeasured where applicable. Old Gemini reports remain historical results; run a fresh evaluation to get Ollama results.

Native structured-output API reference: https://ollama.com/blog/structured-outputs

The quick default uses six single-turn questions, one per main category, and one trial per question. Long conversations and load testing are excluded from this default. Repeatability, memory and full risk coverage are not established by this quick run; REVIEW and unmeasured checks can be correct results. Increase to at least 12 questions and two trials for broader testing.
