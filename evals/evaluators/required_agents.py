"""Validate required workflow agents appeared in the graph execution path."""

from evals.test_cases.schemas import EvalCase, EvalRunResult, EvaluatorResult

# Auditor runs inside the quant_analyst subgraph; parent traces show quant_analyst.
AGENT_PATH_ALIASES: dict[str, list[str]] = {
    "research": ["research"],
    "quant_analyst": ["quant_analyst"],
    "auditor": ["quant_analyst"],
    "analyst": ["analyst"],
}


def evaluate_required_agents(
    case: EvalCase, run: EvalRunResult | None
) -> EvaluatorResult | None:
    if not case.required_agents:
        return None
    if run is None:
        return EvaluatorResult(
            name="required_agents",
            passed=False,
            explanation="No graph run available to validate execution path.",
        )

    path = set(run.execution_path)
    missing: list[str] = []
    for agent in case.required_agents:
        aliases = AGENT_PATH_ALIASES.get(agent, [agent])
        if not any(node in path for node in aliases):
            missing.append(agent)

    passed = not missing
    return EvaluatorResult(
        name="required_agents",
        passed=passed,
        explanation=(
            "All required agents executed."
            if passed
            else f"Missing from execution path: {missing}. Path={run.execution_path}"
        ),
        details={
            "required_agents": case.required_agents,
            "execution_path": run.execution_path,
            "missing": missing,
        },
    )
