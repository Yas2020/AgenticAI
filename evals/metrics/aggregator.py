"""Aggregate offline evaluation metrics."""

from collections import Counter

from evals.test_cases.schemas import CaseEvalResult


def _case_passed(result: CaseEvalResult) -> bool:
    if not result.evaluators:
        return False
    return all(item.passed for item in result.evaluators)


def _metric_from_llm_judge(evaluator, metric_name: str) -> float | None:
    metrics = (evaluator.details or {}).get("metrics") or {}
    entry = metrics.get(metric_name)
    if isinstance(entry, dict) and entry.get("score") is not None:
        return float(entry["score"])
    return None


def aggregate_metrics(results: list[CaseEvalResult]) -> dict:
    total = len(results)
    if total == 0:
        return {
            "success_rate": 0.0,
            "average_faithfulness": 0.0,
            "average_completeness": 0.0,
            "budget_failure_rate": 0.0,
            "tool_usage_rate": 0.0,
            "dag_failure_rate": 0.0,
            "average_latency_ms": 0.0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "estimated_cost_usd": 0.0,
            "top_failure_modes": [],
            "case_count": 0,
        }

    success_count = sum(1 for result in results if _case_passed(result))
    faithfulness_scores: list[float] = []
    completeness_scores: list[float] = []
    tool_usage_count = 0
    dag_failures = 0
    budget_failures = 0
    latencies: list[float] = []
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_cost = 0.0
    failure_counter: Counter[str] = Counter()

    for result in results:
        for evaluator in result.evaluators:
            if evaluator.name == "llm_judge":
                faith = _metric_from_llm_judge(evaluator, "faithfulness")
                comp = _metric_from_llm_judge(evaluator, "completeness")
                if faith is not None:
                    faithfulness_scores.append(faith)
                if comp is not None:
                    completeness_scores.append(comp)
                if not evaluator.passed:
                    failure_counter["LLM judge failure"] += 1
            if evaluator.name == "deterministic:dag_valid" and not evaluator.passed:
                dag_failures += 1
                failure_counter["Invalid dependency graph"] += 1
            if evaluator.name == "deterministic:query_valid" and not evaluator.passed:
                if result.case.category == "ambiguous":
                    failure_counter["Weak reasoning on ambiguous inputs"] += 1
                elif result.case.category == "adversarial":
                    failure_counter["Unsafe or irrelevant input"] += 1
            if evaluator.name == "budgets" and not evaluator.passed:
                budget_failures += 1
                failure_counter["Budget exceeded"] += 1
            if evaluator.name == "resilience" and not evaluator.passed:
                failure_counter["Resilience failure"] += 1
            if evaluator.name == "required_agents" and not evaluator.passed:
                failure_counter["Missing required agents"] += 1
            if not evaluator.passed and evaluator.explanation:
                failure_counter[evaluator.name] += 1

        for mode in result.failure_modes:
            failure_counter[mode] += 1

        if result.run:
            if result.run.tool_call_count > 0:
                tool_usage_count += 1
            latencies.append(result.run.latency_ms)
            token_usage["input_tokens"] += result.run.token_usage.input_tokens
            token_usage["output_tokens"] += result.run.token_usage.output_tokens
            token_usage["total_tokens"] += result.run.token_usage.total_tokens
            total_cost += result.run.estimated_cost_usd

    top_failure_modes = [name for name, _ in failure_counter.most_common(5)]

    return {
        "success_rate": round(success_count / total, 4),
        "average_faithfulness": round(
            sum(faithfulness_scores) / max(len(faithfulness_scores), 1), 3
        ),
        "average_completeness": round(
            sum(completeness_scores) / max(len(completeness_scores), 1), 3
        ),
        "budget_failure_rate": round(budget_failures / total, 4),
        "tool_usage_rate": round(tool_usage_count / total, 4),
        "dag_failure_rate": round(dag_failures / total, 4),
        "average_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
        "token_usage": token_usage,
        "estimated_cost_usd": round(total_cost, 4),
        "top_failure_modes": top_failure_modes,
        "case_count": total,
    }
