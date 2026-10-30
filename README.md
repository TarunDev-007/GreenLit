# GreenLit: real application evaluation

GreenLit evaluates a user's LLM/chatbot, RAG application, or tool-using agent through its live API. The configured model plans tests, generates questions and expected behaviour, and judges actual responses. LangGraph coordinates the workflow, MCP exposes testing tools, and Python measures operations and applies release rules.

## Setup: Gemini only

Install Python 3.11 or newer, then open a terminal in the cloned GreenLit folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m greenlit.cli configure-gemini
.\.venv\Scripts\python.exe -m greenlit.cli check-model
.\.venv\Scripts\python.exe -m greenlit.cli serve
```

Enter your Gemini API key at the hidden terminal prompt. Alternatively, double-click **Set Gemini Key.cmd**, then **Start GreenLit.cmd**. The ignored `.env` file stores your key locally; `.env.example` contains no credentials. Existing target settings and saved reports are preserved.

On macOS/Linux, create and activate a Python environment, install `requirements.lock.txt`, copy `.env.example` to `.env`, and use `python` for the module commands above.

Open http://127.0.0.1:8765. Describe the application, supply its live API URL and matching request/answer paths, and upload reference information where relevant. Choose LLM/chatbot, RAG or agent. Click **Check model** to verify a real Gemini response, then **Run evaluation**.

All model inference uses Google's native Gemini `generateContent` API: planning, question generation, answer judging, and the included math chatbot. Ollama and OpenAI inference are removed; there is no provider fallback. LangGraph orchestrates the stages and MCP exposes testing tools. Python measures latency/errors, applies release rules, compares runs and generates evidence-linked improvement guidance.

The default Gemini model is `gemini-3.5-flash-lite`. To choose another accessible Gemini model, use `configure-gemini --model MODEL_ID`. An optional `JUDGE_MODEL` must also be a Gemini model ID. Requests use a JSON response schema and local Pydantic validation; incomplete, blocked, malformed or quota-limited responses are errors, never passing scores. See [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).

Defaults are 24 questions and two trials. For a small first check, select six questions, one trial and leave load testing off. A quick run does not establish full risk coverage or repeatability. More questions and trials make additional API requests. Cloud calls require internet and model access; quotas and charges depend on the key's project and model. See [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing).

For a live practice target, start **Start Math Chatbot.cmd** in another terminal, then use `http://127.0.0.1:9000/chat`, answer path `answer`, and `greenlit/examples/math-reference.md`. This separate chatbot also uses Gemini and consumes API quota. See [MATH-CHATBOT-TESTING.md](MATH-CHATBOT-TESTING.md). When target and judge share a model, errors can correlate; review judgments against independent reference answers.

Production demo routes and fixture mode are disabled. Bundled deterministic fixtures are used only by automated tests via `GREENLIT_TEST_FIXTURES=1`; do not enable it for real evaluations. Historical reports retain their original provider metadata. Changing evaluators starts a new comparison series.

## Regression and human approval

Every run is saved. GreenLit automatically reuses the latest comparable run's questions for the same API and blocks release on any quality-score decrease. Reports include failed-case prompt/model suggestions, editable human judgments, an audit trail, and an explicit final approval. Corrections recalculate the report and invalidate prior approvals. Suggestions do not add LLM calls or change the target automatically.

See [REGRESSION-AND-REVIEW.md](REGRESSION-AND-REVIEW.md) for the UI steps, comparison rules and API. The calculated recommendation is separate from final approval; automated consumers must check the approval state. Reviewer names are local audit labels, not authenticated identities.

## User workflow

1. Describe the application and supply its live API connection.
2. Upload source documents for RAG, or paste reference facts/acceptance requirements. Plain LLM testing can start with purpose alone, but factual accuracy remains unverified without evidence.
3. The configured model produces a domain-specific plan and fresh test cases covering the requested risk catalog.
4. MCP calls the actual target API, captures responses, retrieval evidence, citations, tool traces, tokens and timings when available.
5. The configured model scores quality and safety against evidence; Python measures operations.
6. GreenLit returns GO/REVIEW/NO-GO with every measured, unmeasured, failed or inapplicable risk listed.

Mandatory checks use a controlled sequence. The configured model cannot silently skip gates, change target addresses or alter release thresholds. Errors and missing evidence never become passing judgments.

## Eval-risk catalog

Application quality: correctness, accuracy, relevance, completeness, instruction following; RAG context relevance, retrieval recall, faithfulness, citation accuracy; tool selection, parameter correctness, task completion, error recovery; conversation context retention and clarification behaviour.

Safety: toxicity, harmful content, bias, PII leakage and jailbreak/prompt leakage resistance. These use generated synthetic scenarios and model judgments, not a safety certification. A scenario must actually test a risk to count toward its coverage.

Operations: target end-to-end latency, estimated target cost per request, token consumption, failure rate, and latency under bounded load. Evaluator generation latency is separate from target latency. Costs require returned token counts and configured target prices; they exclude evaluator charges. Token consumption is a proxy, not proof of optimal token efficiency.

Full risk coverage is required by default. Applicable risks with insufficient evidence or scenarios force REVIEW. Failed measured risk checks block release. RAG/agent-only checks are marked not_applicable for plain chatbots. Inspect the JSON/Markdown report for evidence and limitations.

## Connect any JSON POST API

Default request: `{"message":"{{question}}"}`. Default answer path: `answer`.

For chat-completions-compatible targets use a request body with `messages` and answer path `choices.0.message.content`. Set the target model in your template if required. Authentication uses an environment variable such as CHATBOT_API_KEY; enter the variable name, not its value, in the dashboard. Add the target's exact hostname to ALLOWED_TARGET_HOSTS in `.env`, then reload/restart. Redirects are disabled. Browser-only and arbitrary-code-upload targets are not supported.

### RAG evidence

A RAG API should return its real retrieved passages, for example:

```json
{"answer":"The library closes at 8 PM.","retrieved_contexts":[{"id":"policy-1","text":"The library closes at 8 PM."}],"citations":["policy-1"]}
```

Field paths are configurable. Contexts can be strings or objects with text/page_content/content. Citation checks need actual citations and source metadata. Without retrieved passages the system can check answers against reference text, but it cannot claim retrieval-groundedness.

Retrieval recall currently uses generated, exact source quotes as expected passages and computes their coverage in retrieved text. This is an explicitly labelled source-quote recall proxy, not human-labelled passage recall. Document paraphrases may lower this proxy even when semantically relevant; use a labelled benchmark for research-grade retrieval recall.

### Agent evidence

An agent API should return `tool_calls` containing tool names, arguments and results. Configure its trace field path. Secret-like trace fields are redacted. The configured model judges tool selection, parameters, task completion and recovery only where the scenario and trace provide evidence. A final textual claim is not proof that an action ran. GreenLit does not independently instrument or verify external side effects.

### Multi-turn

To test memory/clarification, configure the optional conversation request body, for example:

```json
{"messages":"{{messages}}"}
```

The placeholder is replaced by the actual user/assistant message array, not a string. You can include other fixed fields, such as your model name. Model-generated cases can contain 2–3 user turns; GreenLit sends the history back to the target for each turn and evaluates the recorded conversation. With no compatible history template these checks are incomplete. Session-ID-only APIs need an additional adapter. Usage aggregation and RAG/agent traces across conversation turns are currently unmeasured, so those combined checks cannot receive a full GO.

### Load

Enable the dashboard checkbox only for a test API where replayed requests, extra provider charges and repeated agent actions are acceptable. The dashboard runs 6 extra requests with 2 concurrent calls. The API/schema supports at most 20 requests and concurrency 5. No load test runs unless explicitly enabled. Unmeasured load coverage is displayed. This is a small bounded load check, not a capacity benchmark.

## Reference uploads

PDF, TXT and Markdown; maximum 10 MB, 100 PDF pages and 50,000 extracted characters. Scanned PDFs need OCR before upload. Encrypted PDFs are rejected. Source content, conversations and traces are untrusted data in model prompts. Reference text and target responses are sent to your configured provider (Google Gemini); use material you are authorized to send.

## MCP tools

`plan_evaluation`, `generate_test_questions`, `run_chatbot_tests`, `evaluate_answers`, `compare_baseline`, `measure_load`, `generate_report`.

## Validation and limits

Run `.\.venv\Scripts\python.exe -m pytest` to check the implementation. Automated tests use isolated local target APIs and intercepted provider responses. These tests verify implementation behaviour; they do not validate a client's application or guarantee judge accuracy. A user's actual API must be connected before its quality can be assessed.

The local dashboard has no multi-user authentication. It is bound to loopback. Keys are not stored in reports. Dataset/configuration hashes and model/package metadata are included for comparison. New generations are not guaranteed repeatable. Jobs are serial and interrupted runs are marked failed.

Optional Gemini references: https://ai.google.dev/api/generate-content and https://ai.google.dev/gemini-api/docs/generate-content/thinking
