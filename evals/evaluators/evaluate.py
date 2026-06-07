"""Orchestrate all evaluators for a single benchmark case."""

from evals.evaluators.budgets import evaluate_budgets
from evals.evaluators.deterministic import evaluate_deterministic_checks
from evals.evaluators.llm_judge import evaluate_llm_judge
from evals.evaluators.required_agents import evaluate_required_agents
from evals.evaluators.resilience import evaluate_resilience
from evals.test_cases.schemas import CaseEvalResult, EvalCase, EvalRunResult, EvaluatorResult


def _append(result: EvaluatorResult | None, bucket: list[EvaluatorResult]) -> None:
    if result is not None:
        bucket.append(result)


def evaluate_case(case: EvalCase, run: EvalRunResult | None = None) -> CaseEvalResult:
    evaluators: list[EvaluatorResult] = []

    evaluators.extend(evaluate_deterministic_checks(case, run))
    _append(evaluate_required_agents(case, run), evaluators)

    if run is not None:
        _append(evaluate_budgets(case, run), evaluators)
        _append(evaluate_resilience(case, run), evaluators)
        _append(evaluate_llm_judge(case, run), evaluators)

    failure_modes = [
        item.explanation for item in evaluators if not item.passed and item.explanation
    ]
    return CaseEvalResult(
        case=case,
        run=run,
        evaluators=evaluators,
        failure_modes=failure_modes,
    )
