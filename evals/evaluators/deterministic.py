"""Deterministic case checks — no LLM required."""

import re

from app.schemas.task import Task
from app.subgraphs.orchestration.plan_validator import collect_plan_errors
from evals.test_cases.schemas import DeterministicCheck, EvalCase, EvalRunResult, EvaluatorResult

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"jailbreak",
    r"<\s*script",
    r"drop\s+table",
    r"system\s+override",
    r"disable\s+safety",
]

AMBIGUOUS_PATTERNS = [
    r"good investment",
    r"chip company",
    r"tell me if it is",
    r"strong ai demand",
]


def _rule_based_query_valid(query: str) -> tuple[bool, str]:
    lowered = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"Matched unsafe pattern: {pattern}"
    for pattern in AMBIGUOUS_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"Ambiguous query pattern: {pattern}"
    if len(query.strip()) < 8:
        return False, "Query too short or malformed."
    return True, "Rule-based checks passed."


def _plan_tasks(case: EvalCase, run: EvalRunResult | None) -> list[Task]:
    if case.synthetic_plan:
        return [Task(**item) for item in case.synthetic_plan]
    if run and run.plan:
        return run.plan
    return []


def _execution_errors(tasks: list[Task]) -> list[str]:
    failed = [t.id for t in tasks if t.status == "failed"]
    if failed:
        return [f"Downstream execution failed for task(s) {failed}."]
    return []


def _check_query_valid(
    case: EvalCase, run: EvalRunResult | None, check: DeterministicCheck
) -> EvaluatorResult:
    expected = True if check.expected is None else check.expected
    rule_pass, rule_reason = _rule_based_query_valid(case.input_query)

    if run is not None and run.is_query_valid is not None:
        actual = run.is_query_valid and rule_pass
    else:
        actual = rule_pass

    passed = actual == expected
    return EvaluatorResult(
        name="deterministic:query_valid",
        passed=passed,
        explanation=(
            f"Expected query_valid={expected}, observed={actual}. {rule_reason}"
        ),
        details={"expected": expected, "rule_pass": rule_pass, "observed": actual},
    )


def _check_dag_valid(
    case: EvalCase, run: EvalRunResult | None, check: DeterministicCheck
) -> EvaluatorResult:
    expected = True if check.expected is None else check.expected
    tasks = _plan_tasks(case, run)
    if not tasks:
        return EvaluatorResult(
            name="deterministic:dag_valid",
            passed=expected is False,
            explanation="No plan available for DAG evaluation.",
            details={"expected": expected, "errors": ["no plan"]},
        )

    errors = collect_plan_errors(tasks)
    if run is not None and expected is not False:
        errors.extend(_execution_errors(tasks))

    dag_valid = len(errors) == 0
    passed = dag_valid == expected
    return EvaluatorResult(
        name="deterministic:dag_valid",
        passed=passed,
        score=0.0 if dag_valid else 1.0,
        explanation="; ".join(errors) if errors else "DAG is valid.",
        details={"expected": expected, "dag_valid": dag_valid, "errors": errors},
    )


def _check_plan_has_agent(
    case: EvalCase, run: EvalRunResult | None, check: DeterministicCheck
) -> EvaluatorResult:
    agent = check.agent
    if not agent:
        return EvaluatorResult(
            name="deterministic:plan_has_agent",
            passed=False,
            explanation="Missing agent field on plan_has_agent check.",
        )

    tasks = _plan_tasks(case, run)
    agents = {task.agent for task in tasks}
    passed = agent in agents
    return EvaluatorResult(
        name=f"deterministic:plan_has_agent:{agent}",
        passed=passed,
        explanation=(
            f"Plan {'includes' if passed else 'missing'} agent '{agent}'. "
            f"Found: {sorted(agents)}"
        ),
        details={"agent": agent, "plan_agents": sorted(agents)},
    )


def _target_text(run: EvalRunResult | None, target: str | None) -> str:
    if target == "final_report" and run and run.final_response:
        return run.final_response
    return ""


def _check_regex(
    case: EvalCase, run: EvalRunResult | None, check: DeterministicCheck
) -> EvaluatorResult:
    if not check.pattern:
        return EvaluatorResult(
            name="deterministic:regex",
            passed=False,
            explanation="Missing pattern on regex check.",
        )
    text = _target_text(run, check.target)
    if not text:
        return EvaluatorResult(
            name="deterministic:regex",
            passed=False,
            explanation="No final report available for regex check.",
        )
    passed = re.search(check.pattern, text) is not None
    return EvaluatorResult(
        name="deterministic:regex",
        passed=passed,
        explanation=(
            f"Pattern {'matched' if passed else 'not matched'}: {check.pattern}"
        ),
        details={"pattern": check.pattern, "target": check.target},
    )


def _check_min_length(
    case: EvalCase, run: EvalRunResult | None, check: DeterministicCheck
) -> EvaluatorResult:
    min_chars = check.min_chars or 0
    text = _target_text(run, check.target or "final_report")
    length = len(text)
    passed = length >= min_chars
    return EvaluatorResult(
        name="deterministic:min_length",
        passed=passed,
        score=float(length),
        explanation=f"Report length={length}, required>={min_chars}.",
        details={"length": length, "min_chars": min_chars},
    )


_CHECK_HANDLERS = {
    "query_valid": _check_query_valid,
    "dag_valid": _check_dag_valid,
    "plan_has_agent": _check_plan_has_agent,
    "regex": _check_regex,
    "min_length": _check_min_length,
}


def evaluate_deterministic_checks(
    case: EvalCase, run: EvalRunResult | None
) -> list[EvaluatorResult]:
    results: list[EvaluatorResult] = []
    for check in case.checks.deterministic:
        handler = _CHECK_HANDLERS.get(check.type)
        if handler is None:
            results.append(
                EvaluatorResult(
                    name=f"deterministic:{check.type}",
                    passed=False,
                    explanation=f"Unknown deterministic check type: {check.type}",
                )
            )
            continue
        results.append(handler(case, run, check))
    return results
