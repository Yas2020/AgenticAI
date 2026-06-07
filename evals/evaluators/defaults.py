"""Category defaults for LLM thresholds, graph-run budgets, and resilience."""

from evals.test_cases.schemas import (
    BudgetSpec,
    EvalCase,
    LLMMetric,
    LLMMetricCheck,
    ResilienceExpect,
    ResilienceSpec,
)

ALL_LLM_METRICS: list[LLMMetric] = [
    "faithfulness",
    "citation_fidelity",
    "consistency",
    "completeness",
]

# Applied to every run_full_graph case unless overridden per case.
GLOBAL_GRAPH_BUDGET = BudgetSpec(
    max_cost_usd=0.12,
    max_input_tokens=100_000,
    max_output_tokens=16_000,
    max_latency_ms=120_000,
)

DEFAULT_RESILIENCE = ResilienceSpec(
    mode="natural",
    expect=[
        ResilienceExpect(type="graph_completed"),
        ResilienceExpect(type="no_infinite_retry", max_graph_steps=50),
        ResilienceExpect(type="min_tool_calls", min_tool_calls=1),
    ],
)

DEFAULT_LLM_THRESHOLDS: dict[str, dict[LLMMetric, float]] = {
    "simple_task": {
        "faithfulness": 0.7,
        "citation_fidelity": 0.6,
        "consistency": 0.7,
        "completeness": 0.7,
    },
    "multi_hop": {
        "faithfulness": 0.7,
        "citation_fidelity": 0.6,
        "consistency": 0.7,
        "completeness": 0.7,
    },
    "missing_data": {
        "faithfulness": 0.75,
        "citation_fidelity": 0.5,
        "consistency": 0.7,
        "completeness": 0.65,
    },
    "hallucination_prone": {
        "faithfulness": 0.8,
        "citation_fidelity": 0.6,
        "consistency": 0.7,
        "completeness": 0.6,
    },
}

DEFAULT_BUDGETS: dict[str, BudgetSpec] = {
    "simple_task": BudgetSpec(
        max_cost_usd=0.08,
        max_input_tokens=80_000,
        max_output_tokens=12_000,
        max_latency_ms=90_000,
    ),
    "multi_hop": BudgetSpec(
        max_cost_usd=0.15,
        max_input_tokens=120_000,
        max_output_tokens=20_000,
        max_latency_ms=120_000,
    ),
    "missing_data": BudgetSpec(
        max_cost_usd=0.08,
        max_input_tokens=80_000,
        max_output_tokens=12_000,
        max_latency_ms=90_000,
    ),
    "hallucination_prone": BudgetSpec(
        max_cost_usd=0.08,
        max_input_tokens=80_000,
        max_output_tokens=12_000,
        max_latency_ms=90_000,
    ),
}


def resolve_llm_checks(case: EvalCase) -> list[LLMMetricCheck]:
    if case.checks.llm:
        return case.checks.llm
    defaults = DEFAULT_LLM_THRESHOLDS.get(case.category)
    if not defaults:
        return []
    return [
        LLMMetricCheck(metric=metric, min_score=threshold)
        for metric, threshold in defaults.items()
    ]


def resolve_budgets(case: EvalCase) -> BudgetSpec | None:
    if not case.run_full_graph:
        return None
    base = GLOBAL_GRAPH_BUDGET
    category = DEFAULT_BUDGETS.get(case.category)
    if category:
        base = BudgetSpec(
            max_cost_usd=category.max_cost_usd or base.max_cost_usd,
            max_input_tokens=category.max_input_tokens or base.max_input_tokens,
            max_output_tokens=category.max_output_tokens or base.max_output_tokens,
            max_latency_ms=category.max_latency_ms or base.max_latency_ms,
        )
    overrides = case.budgets
    if overrides is None:
        return base
    return BudgetSpec(
        max_cost_usd=overrides.max_cost_usd or base.max_cost_usd,
        max_input_tokens=overrides.max_input_tokens or base.max_input_tokens,
        max_output_tokens=overrides.max_output_tokens or base.max_output_tokens,
        max_latency_ms=overrides.max_latency_ms or base.max_latency_ms,
    )


def resolve_resilience(case: EvalCase) -> ResilienceSpec | None:
    if not case.run_full_graph:
        return None
    if case.resilience and case.resilience.expect:
        return case.resilience
    return DEFAULT_RESILIENCE
