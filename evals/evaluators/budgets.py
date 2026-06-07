"""Graph-run budget rails — cost, tokens, latency."""

from evals.evaluators.defaults import resolve_budgets
from evals.test_cases.schemas import EvalCase, EvalRunResult, EvaluatorResult


def evaluate_budgets(case: EvalCase, run: EvalRunResult | None) -> EvaluatorResult | None:
    budgets = resolve_budgets(case)
    if budgets is None or run is None:
        return None

    violations: list[str] = []
    details: dict[str, float | int] = {
        "estimated_cost_usd": run.estimated_cost_usd,
        "input_tokens": run.token_usage.input_tokens,
        "output_tokens": run.token_usage.output_tokens,
        "latency_ms": run.latency_ms,
    }

    if budgets.max_cost_usd is not None and run.estimated_cost_usd > budgets.max_cost_usd:
        violations.append(
            f"cost ${run.estimated_cost_usd:.4f} > max ${budgets.max_cost_usd:.4f}"
        )
    if (
        budgets.max_input_tokens is not None
        and run.token_usage.input_tokens > budgets.max_input_tokens
    ):
        violations.append(
            f"input tokens {run.token_usage.input_tokens} > max {budgets.max_input_tokens}"
        )
    if (
        budgets.max_output_tokens is not None
        and run.token_usage.output_tokens > budgets.max_output_tokens
    ):
        violations.append(
            f"output tokens {run.token_usage.output_tokens} > max {budgets.max_output_tokens}"
        )
    if budgets.max_latency_ms is not None and run.latency_ms > budgets.max_latency_ms:
        violations.append(
            f"latency {run.latency_ms:.0f}ms > max {budgets.max_latency_ms:.0f}ms"
        )

    passed = not violations
    return EvaluatorResult(
        name="budgets",
        passed=passed,
        explanation="Within budget rails." if passed else "; ".join(violations),
        details={
            **details,
            "limits": budgets.model_dump(exclude_none=True),
            "violations": violations,
        },
    )
