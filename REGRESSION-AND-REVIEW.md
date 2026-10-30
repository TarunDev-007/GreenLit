# Regression, improvements and human review

## Normal workflow

1. Run an evaluation in GreenLit. Every run is stored in the local SQLite history, including failed/interrupted runs.
2. For the same application API, the latest completed run is selected automatically. If the purpose, reference, evaluator, test count, trials, evidence mapping and release criteria are comparable, its questions and plan are reused. An explicit approved baseline can also be selected.
3. Any quality-score decrease in a comparable run gives **NO-GO**. The old regression tolerance setting is retained only for configuration compatibility; it does not permit score declines.
4. Open **What to improve** for failed/uncertain cases. Suggestions link to case IDs and findings, describe prompt and model/system changes, and recommend rerunning the saved dataset. They are evidence-linked guidance rather than LLM-generated or verified fixes. They do not change the target prompt/model automatically or add Gemini calls.
5. Enter a reviewer name under **Human review & final approval**. Expand a response and **Review this judgment**. Edit the verdict, scores, explanation and relevant risk checks, provide a reason, then save.
6. The report recalculates quality, risk coverage, regressions and the recommendation. The original AI assessment remains available. **Restore AI judgment** reverses an override while retaining its audit history.
7. Provide an approval reason, then choose **Approve final GO** or **Approve final NO-GO**. GO is enabled only when the calculated recommendation is GO. NO-GO can hold any release, including one the software recommends GO for.

Changing a judgment after approval resets approval to pending and removes the saved-baseline flag. A reviewer must approve the revised report again. Saving a manual baseline requires a human-approved GO report.

## What comparison means

- Automatic application identity is the exact target API URL and application type. Different URLs can be compared by explicitly selecting an approved baseline.
- Comparison uses the latest completed run for that application, including a run that failed quality checks. Failed/interrupted requests with no completed report are retained in history but are not numeric comparison baselines.
- The first run has no previous score. If the latest run uses incompatible inputs/settings, the report explains the mismatch, generates a fresh dataset and requires REVIEW rather than presenting an unfair comparison as passing.
- A snapshot of the comparison score and judgments, including its revision, is frozen for each new run. Later edits to an old report do not silently change existing regression results.
- Model changes in the application under test can be evaluated against saved questions. A change to the **evaluator** starts a new comparison series because its scores may not be comparable.

## Review integrity and limits

Original machine reports and append-only application audit events are stored separately. Events record the reviewer label, reason, time, case/trial, old/new judgment and revision. Revision checks prevent two reviewers from silently overwriting each other's work. The audit is not cryptographically tamper-proof.

Reviewers can grade an actual response when the AI judge failed, but cannot manufacture an answer for a failed target request. Missing RAG retrieval/tool evidence, observed latency failures, incomplete coverage and score regressions are not cleared by simply approving GO. Existing risk checks must be explicitly assessed rather than silently removed.

The app remains local and has no multi-user authentication. A reviewer name is an audit label, not a verified identity or role. A shared deployment needs authentication and authorization before relying on these approvals for access control. No deployment or external release happens when a report is approved.

Historical reports remain accessible. Legacy reports that compared a baseline without storing a frozen snapshot must be rerun before review, so corrections cannot silently discard their old regression gates.

## API

- `POST /api/runs` — `compare_previous` defaults to true; an explicit `baseline_id` takes precedence. Advanced clients can explicitly set `compare_previous` to false.
- `POST /api/runs/{id}/overrides` — reviewer, reason, current revision, case_id, trial and complete Judgment.
- `POST /api/runs/{id}/overrides/reset` — reviewer, reason, current revision, case_id and trial.
- `POST /api/runs/{id}/approval` — reviewer, reason, current revision and verdict (`GO` or `NO-GO`).
- Stale revisions return HTTP 409. Missing reasons or invalid grades return HTTP 422.
- JSON and Markdown exports include effective scores, original automated recommendation, suggestions, human overrides, approval state and review audit history.

`decision.verdict` is the calculated recommendation. An approved final GO requires **both** `approval.status == "approved"` and `approval.verdict == "GO"`. CLI runs leave human approval pending and do not return a successful release exit code for an unapproved GO.
