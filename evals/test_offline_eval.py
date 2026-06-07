"""Offline evaluation pytest entrypoint."""

import os

import pytest

from app.services.observability.langfuse_tracer import create_eval_trace
from evals.evaluators.budgets import evaluate_budgets
from evals.evaluators.defaults import resolve_budgets
from evals.evaluators.deterministic import evaluate_deterministic_checks
from evals.evaluators.evaluate import evaluate_case
from evals.evaluators.llm_judge import evaluate_llm_judge
from evals.evaluators.required_agents import evaluate_required_agents
from evals.evaluators.resilience import evaluate_resilience
from evals.test_cases.rate_limit import run_with_retry, sleep_between_cases
from evals.test_cases.runner import run_case
from evals.test_cases.schemas import (
    CaseEvalResult,
    ChecksSpec,
    DeterministicCheck,
    EvalCase,
    EvalRunResult,
    LLMMetricCheck,
    TokenUsage,
)


@pytest.mark.unit
def test_query_valid_check():
    case = EvalCase(
        id="unit-query-001",
        category="ambiguous",
        input_query="Analyze the chip company with strong AI demand.",
        run_full_graph=False,
        checks=ChecksSpec(
            deterministic=[DeterministicCheck(type="query_valid", expected=False)]
        ),
    )
    results = evaluate_deterministic_checks(case, None)
    assert len(results) == 1
    assert results[0].name == "deterministic:query_valid"
    assert results[0].passed


@pytest.mark.unit
def test_required_agents_check():
    case = EvalCase(
        id="unit-agents-001",
        category="multi_hop",
        input_query="test",
        required_agents=["research", "analyst"],
    )
    run = EvalRunResult(
        case_id=case.id,
        category=case.category,
        execution_path=["query_validator", "research", "analyst"],
    )
    result = evaluate_required_agents(case, run)
    assert result is not None
    assert result.passed


@pytest.mark.unit
def test_budgets_within_global_limits():
    case = EvalCase(
        id="unit-budget-pass",
        category="simple_task",
        input_query="test",
        run_full_graph=True,
    )
    run = EvalRunResult(
        case_id=case.id,
        category=case.category,
        estimated_cost_usd=0.05,
        latency_ms=30_000,
        token_usage=TokenUsage(input_tokens=10_000, output_tokens=2_000, total_tokens=12_000),
    )
    result = evaluate_budgets(case, run)
    assert result is not None
    assert result.name == "budgets"
    assert result.passed
    limits = resolve_budgets(case)
    assert limits is not None
    assert result.details["limits"]["max_cost_usd"] == limits.max_cost_usd


@pytest.mark.unit
def test_budgets_exceeded():
    case = EvalCase(
        id="unit-budget-fail",
        category="simple_task",
        input_query="test",
        run_full_graph=True,
    )
    run = EvalRunResult(
        case_id=case.id,
        category=case.category,
        estimated_cost_usd=0.50,
        latency_ms=30_000,
        token_usage=TokenUsage(input_tokens=10_000, output_tokens=2_000, total_tokens=12_000),
    )
    result = evaluate_budgets(case, run)
    assert result is not None
    assert not result.passed
    assert "cost" in result.explanation.lower()


@pytest.mark.unit
def test_resilience_default_infra_checks():
    case = EvalCase(
        id="unit-resilience-pass",
        category="simple_task",
        input_query="test",
        run_full_graph=True,
    )
    run = EvalRunResult(
        case_id=case.id,
        category=case.category,
        success=True,
        execution_path=["query_validator", "research", "analyst"],
        tool_call_count=2,
    )
    result = evaluate_resilience(case, run)
    assert result is not None
    assert result.passed
    check_types = {item["type"] for item in result.details["checks"]}
    assert "graph_completed" in check_types
    assert "no_infinite_retry" in check_types
    assert "min_tool_calls" in check_types


@pytest.mark.unit
def test_resilience_catches_runaway_steps():
    case = EvalCase(
        id="unit-resilience-fail",
        category="multi_hop",
        input_query="test",
        run_full_graph=True,
    )
    run = EvalRunResult(
        case_id=case.id,
        category=case.category,
        success=True,
        execution_path=["quant_analyst"] * 60,
        tool_call_count=5,
    )
    result = evaluate_resilience(case, run)
    assert result is not None
    assert not result.passed
    assert "no_infinite_retry" in result.explanation


@pytest.mark.unit
def test_resilience_catches_missing_mcp_tools():
    case = EvalCase(
        id="unit-resilience-mcp",
        category="simple_task",
        input_query="test",
        run_full_graph=True,
    )
    run = EvalRunResult(
        case_id=case.id,
        category=case.category,
        success=True,
        execution_path=["query_validator", "analyst"],
        tool_call_count=0,
    )
    result = evaluate_resilience(case, run)
    assert result is not None
    assert not result.passed
    assert "min_tool_calls" in result.explanation


@pytest.mark.unit
def test_llm_judge_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    case = EvalCase(
        id="unit-judge-001",
        category="simple_task",
        input_query="test",
        checks=ChecksSpec(
            llm=[LLMMetricCheck(metric="faithfulness", min_score=0.7)]
        ),
    )
    run = EvalRunResult(
        case_id=case.id,
        category=case.category,
        final_response="Example report.",
        artifacts=[
            {
                "artifact_type": "web_research",
                "source": "research",
                "content": {"revenue": "26.1B"},
            }
        ],
    )
    result = evaluate_llm_judge(case, run)
    assert result is not None
    assert result.passed
    assert "Skipped" in result.explanation


@pytest.mark.unit
def test_unit_evaluators(code_cases, evaluate_case, finalize_report, request):
    results = [evaluate_case(case) for case in code_cases]
    metrics = finalize_report(results, request)
    assert metrics["case_count"] == len(code_cases)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_eval_pipeline(
    integration_cases,
    eval_graph,
    mcp_lifecycle,
    evaluate_case,
    finalize_report,
    request,
):
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for integration evals.")

    delay = request.config.getoption("--eval-case-delay")
    results: list[CaseEvalResult] = []

    for index, case in enumerate(integration_cases):
        if index > 0:
            await sleep_between_cases(delay)

        async def _run_case(current_case=case):
            from evals.langfuse_export import record_case_eval

            trace = create_eval_trace(current_case.id, current_case.category)
            config = trace.run_config()
            run = await run_case(eval_graph, current_case, config=config)
            case_eval = evaluate_case(current_case, run)
            record_case_eval(trace.trace_id, current_case, case_eval, run)
            trace.end(
                output={
                    "passed": case_eval.evaluators
                    and all(e.passed for e in case_eval.evaluators),
                    "eval_case_passed": all(e.passed for e in case_eval.evaluators),
                    "execution_path": run.execution_path,
                    "latency_ms": run.latency_ms,
                    "tool_call_count": run.tool_call_count,
                    "estimated_cost_usd": run.estimated_cost_usd,
                    "token_usage": run.token_usage.model_dump(),
                    "final_response": run.final_response,
                    "evaluators": [e.model_dump() for e in case_eval.evaluators],
                },
            )
            return run, case_eval

        _run, case_eval = await run_with_retry(_run_case)
        results.append(case_eval)

    metrics = finalize_report(results, request)
    assert metrics["case_count"] == len(integration_cases)
