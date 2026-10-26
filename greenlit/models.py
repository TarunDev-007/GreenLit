import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["correctness", "groundedness", "unknown", "injection", "safety", "over_refusal"]
CATEGORIES = ["correctness", "groundedness", "unknown", "injection", "safety", "over_refusal"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Target(StrictModel):
    application_type: Literal['chatbot','rag','agent'] = 'chatbot'
    conversation_request_template: dict | None = None
    citations_path: str = Field(default='citations', max_length=150)
    contexts_path: str = Field(default='retrieved_contexts', max_length=150)
    tool_trace_path: str = Field(default='tool_calls', max_length=150)
    url: str = Field(min_length=8, max_length=2048)
    request_template: dict = Field(default_factory=lambda: {"message": "{{question}}"})
    response_path: str = Field(default="answer", max_length=150)
    # A secret VARIABLE NAME, never a secret value. Only CHATBOT_* variables allowed.
    api_key_env: str = Field(default="", pattern=r"^(CHATBOT_[A-Z0-9_]+)?$")
    auth_header: Literal["Authorization", "X-API-Key"] = "Authorization"
    timeout_seconds: float = Field(default=20, ge=1, le=120)

    @model_validator(mode="after")
    def has_placeholder(self):
        import json
        if "{{question}}" not in json.dumps(self.request_template):
            raise ValueError("Request template must contain {{question}}")
        return self


class LoadTest(StrictModel):
    enabled: bool = False
    requests: int = Field(default=6,ge=2,le=20)
    concurrency: int = Field(default=2,ge=1,le=5)


class RiskMetric(StrictModel):
    status: Literal['pass','fail','not_measured']
    score: float | None = Field(default=None,ge=0,le=1)
    reason: str = Field(min_length=5,max_length=2000)


class Rules(StrictModel):
    require_full_coverage: bool = True
    go_score: float = Field(default=0.8, ge=0, le=1)
    no_go_score: float = Field(default=0.5, ge=0, le=1)
    max_p95_ms: float = Field(default=5000, gt=0, le=120000)
    max_regression_drop: float = Field(default=0, ge=0, le=1)
    min_judge_confidence: float = Field(default=0.7, ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.no_go_score >= self.go_score:
            raise ValueError("NO-GO score must be lower than GO score")
        return self


class RunRequest(StrictModel):
    name: str = Field(default="College FAQ evaluation", min_length=1, max_length=100)
    purpose: str = Field(min_length=10, max_length=4000)
    reference: str = Field(default="", max_length=50000)
    target: Target
    load_test: LoadTest = Field(default_factory=LoadTest)
    test_count: int = Field(default_factory=lambda: 6 if os.getenv('LLM_PROVIDER','').lower()=='ollama' else 24, ge=6, le=60)
    repeats: int = Field(default_factory=lambda: 1 if os.getenv('LLM_PROVIDER','').lower()=='ollama' else 2, ge=1, le=3)
    rules: Rules = Field(default_factory=Rules)
    compare_previous: bool = True
    baseline_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class Case(StrictModel):
    reference_evidence_verified: bool | None = None
    validation_notes: list[str] = Field(default_factory=list,max_length=10)
    risk_tags: list[str] = Field(default_factory=list,max_length=10)
    turns: list[str] = Field(default_factory=list,max_length=3)
    expected_passages: list[str] = Field(default_factory=list,max_length=20)
    id: str = Field(min_length=1, max_length=80)
    category: Category
    question: str = Field(min_length=3, max_length=3000)
    expected_behavior: str = Field(min_length=1, max_length=3000)
    reference_evidence: str = Field(default="", max_length=5000)


class LocalQuestion(StrictModel):
    """Small generation contract for local models; metadata is added by Python."""
    question: str = Field(min_length=3,max_length=3000)
    expected_behavior: str = Field(min_length=1,max_length=3000)
    reference_evidence: str = Field(default='',max_length=5000)
    turns: list[str] = Field(default_factory=list,max_length=3)


class GeneratedCases(StrictModel):
    cases: list[Case]


class RetrievalJudgment(StrictModel):
    context_relevance: float = Field(ge=0, le=1)
    answer_supported_by_context: bool
    reason: str = Field(min_length=5, max_length=2000)


class AgentJudgment(StrictModel):
    tool_use_correctness: float = Field(ge=0, le=1)
    task_success: Literal['pass','fail','unknown']
    reason: str = Field(min_length=5, max_length=2000)


class Judgment(StrictModel):
    risk_metrics: dict[str,RiskMetric] = Field(default_factory=dict)
    retrieval: RetrievalJudgment | None = None
    agent: AgentJudgment | None = None
    verdict: Literal["pass", "fail", "review"]
    correctness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=5, max_length=3000)



class CategoryFocus(StrictModel):
    category: Category
    focus: str = Field(min_length=5, max_length=1500)


class TestPlan(StrictModel):
    objective: str = Field(min_length=5, max_length=2000)
    application_risks: list[str] = Field(min_length=1, max_length=12)
    categories: list[CategoryFocus] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def complete_categories(self):
        if {item.category for item in self.categories} != set(CATEGORIES):
            raise ValueError("Test plan must cover all six required categories exactly once")
        return self


class ReviewIdentity(StrictModel):
    model_config = ConfigDict(extra='forbid',str_strip_whitespace=True)
    reviewer: str = Field(min_length=1,max_length=100)
    reason: str = Field(min_length=5,max_length=2000)
    revision: int = Field(ge=0)


class OverrideRequest(ReviewIdentity):
    case_id: str = Field(min_length=1,max_length=80)
    trial: int = Field(ge=1,le=3)
    judgment: Judgment


class ResetOverrideRequest(ReviewIdentity):
    case_id: str = Field(min_length=1,max_length=80)
    trial: int = Field(ge=1,le=3)


class ApprovalRequest(ReviewIdentity):
    verdict: Literal['GO','NO-GO']
