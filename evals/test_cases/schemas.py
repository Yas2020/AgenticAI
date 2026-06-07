"""Evaluation case schemas."""

from typing import Any, Literal
from pydantic import BaseModel, Field

from app.schemas.task import Task

CaseCategory = Literal[
    "simple_task",
    "multi_hop",
    "ambiguous",
    "adversarial",
    "missing_data",
    "hallucination_prone",
    "dag_edge_case",
    "dependency_conflict",
]

LLMMetric = Literal[
    "faithfulness",
    "citation_fidelity",
    "consistency",
    "completeness",
]


class DeterministicCheck(BaseModel):
    type: Literal[
        "query_valid",
        "dag_valid",
        "plan_has_agent",
        "regex",
        "min_length",
    ]
    expected: bool | None = None
    agent: str | None = None
    target: Literal["final_report"] | None = None
    pattern: str | None = None
    min_chars: int | None = None


class LLMMetricCheck(BaseModel):
    metric: LLMMetric
    min_score: float = Field(ge=0.0, le=1.0)


class ResilienceExpect(BaseModel):
    type: Literal[
        "graph_completed",
        "no_infinite_retry",
        "failed_task_handled",
        "min_tool_calls",
    ]
    max_graph_steps: int | None = None
    min_tool_calls: int | None = None
    agent: str | None = None


class ResilienceSpec(BaseModel):
    mode: Literal["natural", "injected"] = "natural"
    expect: list[ResilienceExpect] = Field(default_factory=list)


class BudgetSpec(BaseModel):
    max_cost_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_latency_ms: float | None = None


class ChecksSpec(BaseModel):
    deterministic: list[DeterministicCheck] = Field(default_factory=list)
    llm: list[LLMMetricCheck] = Field(default_factory=list)


class EvalCase(BaseModel):
    id: str
    category: CaseCategory
    input_query: str
    topic: str = "equity research"
    run_full_graph: bool = True
    checks: ChecksSpec = Field(default_factory=ChecksSpec)
    required_agents: list[str] = Field(default_factory=list)
    resilience: ResilienceSpec | None = None
    budgets: BudgetSpec | None = None
    synthetic_plan: list[dict[str, Any]] | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class EvalRunResult(BaseModel):
    case_id: str
    category: str
    success: bool = False
    latency_ms: float = 0.0
    execution_path: list[str] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[Task] = Field(default_factory=list)
    is_query_valid: bool | None = None
    is_plan_valid: bool | None = None
    tool_call_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    error: str | None = None
    final_response: str | None = None


class EvaluatorResult(BaseModel):
    name: str
    passed: bool
    score: float | None = None
    explanation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class CaseEvalResult(BaseModel):
    case: EvalCase
    run: EvalRunResult | None = None
    evaluators: list[EvaluatorResult] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
