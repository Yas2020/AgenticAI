"""Resilience expectations for natural failure scenarios."""

from evals.evaluators.defaults import resolve_resilience
from evals.test_cases.schemas import EvalCase, EvalRunResult, EvaluatorResult


def evaluate_resilience(case: EvalCase, run: EvalRunResult | None) -> EvaluatorResult | None:
    resilience = resolve_resilience(case)
    if resilience is None:
        return None
    if run is None:
        return EvaluatorResult(
            name="resilience",
            passed=False,
            explanation="No graph run available for resilience checks.",
        )

    failures: list[str] = []
    details: dict = {"checks": [], "mode": resilience.mode}

    for expect in resilience.expect:
        if expect.type == "graph_completed":
            ok = run.success and run.error is None
            if not ok:
                failures.append(f"graph_completed failed: error={run.error}")
            details["checks"].append({"type": expect.type, "passed": ok})

        elif expect.type == "no_infinite_retry":
            limit = expect.max_graph_steps or 50
            steps = len(run.execution_path)
            ok = steps <= limit
            if not ok:
                failures.append(
                    f"no_infinite_retry failed: {steps} steps > max {limit}"
                )
            details["checks"].append(
                {"type": expect.type, "passed": ok, "steps": steps, "limit": limit}
            )

        elif expect.type == "failed_task_handled":
            failed = [t.id for t in run.plan if t.status == "failed"]
            ok = run.success or bool(failed)
            if run.error and not failed:
                ok = False
            if not ok:
                failures.append(
                    "failed_task_handled: graph errored without task failure state"
                )
            details["checks"].append(
                {"type": expect.type, "passed": ok, "failed_tasks": failed}
            )

        elif expect.type == "min_tool_calls":
            minimum = expect.min_tool_calls or 1
            ok = run.tool_call_count >= minimum
            if not ok:
                failures.append(
                    f"min_tool_calls failed: {run.tool_call_count} < {minimum} "
                    "(MCP tools may be down)"
                )
            details["checks"].append(
                {
                    "type": expect.type,
                    "passed": ok,
                    "tool_call_count": run.tool_call_count,
                    "min_tool_calls": minimum,
                }
            )

    passed = not failures
    return EvaluatorResult(
        name="resilience",
        passed=passed,
        explanation="Resilience expectations met." if passed else "; ".join(failures),
        details=details,
    )
