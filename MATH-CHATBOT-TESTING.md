# Test the real math teaching chatbot

This is a separate application whose answers are generated live by the configured model. It uses the Gemini model and API key configured in your local `.env`. It has no programmed answer fixtures. Configure your Gemini key before starting the chatbot.

Start with `Start Math Chatbot.cmd` in your GreenLit project folder, or run `.\.venv\Scripts\python.exe -m greenlit.math_chatbot` from that folder.

In GreenLit:

- Application type: LLM / chatbot (this math assistant is not RAG and does not use tools).
- API URL: http://127.0.0.1:9000/chat
- Answer field path: answer
- Purpose: A patient mathematics teaching assistant that solves school-level mathematics, explains concise steps, follows English/Hindi instructions, asks for missing information and protects private data/instructions.
- Upload `greenlit/examples/math-reference.md` as the reference.
- Request template: `{"message":"{{question}}"}`.
- Multi-turn request template: `{"messages":"{{messages}}"}`.
- Credential variable: empty for this local target; its server uses the Gemini key from `.env`.
- Start with 6 questions and one trial; target timeout 120 seconds.
- For the first run, leave load testing disabled. Enable the bounded load check explicitly when ready for extra requests.

A full-coverage evaluation may correctly return REVIEW for unmeasured risks, missing scenarios, unavailable target pricing, disabled load coverage, or unmeasured multi-turn token totals. No GO result is guaranteed.

API health: http://127.0.0.1:9000/health
Interactive API documentation: http://127.0.0.1:9000/docs

Example single-turn request: `{"message":"Solve 2x + 4 = 12. Give only the answer."}`
Example multi-turn request: `{"messages":[{"role":"user","content":"My name is Riya."},{"role":"assistant","content":"Hello Riya."},{"role":"user","content":"What is my name?"}]}`

The same provider currently powers the target and the evaluator, which can create correlated errors. Use an independent evaluator/provider and external reference labels for stronger research validation. Keep this unauthenticated testing API bound to the local computer.
