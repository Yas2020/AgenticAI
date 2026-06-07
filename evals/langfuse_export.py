"""Push offline eval EvaluatorResult scores and dataset links into Langfuse."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LG_ROOT = ROOT / "lg"
if str(LG_ROOT) not in sys.path:
    sys.path.insert(0, str(LG_ROOT))

from evals.test_cases.schemas import CaseEvalResult, EvalCase, EvalRunResult  # noqa: E402
from app.services.observability.langfuse_tracer import (  # noqa: E402
    get_langfuse_client,
    is_langfuse_enabled,
)

EVAL_DATASET_NAME = os.getenv("LANGFUSE_EVAL_DATASET", "investment_research_benchmark")


def _safe_score_name(name: str) -> str:
    return name.replace(":", "_").replace(" ", "_")[:64]


def _ensure_benchmark_dataset(client) -> None:
    try:
        client.create_dataset(
            name=EVAL_DATASET_NAME,
            description="Offline benchmark cases from evals/datasets/benchmark.jsonl",
        )
    except Exception:
        pass


def _upsert_dataset_item(client, case: EvalCase) -> str | None:
    try:
        item = client.create_dataset_item(
            dataset_name=EVAL_DATASET_NAME,
            id=case.id,
            input={
                "case_id": case.id,
                "category": case.category,
                "query": case.input_query,
                "topic": case.topic,
            },
            expected_output={
                "eval_case_passed": True,
                "notes": "Pass when all evaluators pass (see trace scores).",
            },
            metadata={"run_full_graph": case.run_full_graph},
        )
        if isinstance(item, dict):
            return item.get("id") or case.id
        return getattr(item, "id", None) or case.id
    except TypeError:
        try:
            item = client.create_dataset_item(
                dataset_name=EVAL_DATASET_NAME,
                input={"case_id": case.id, "query": case.input_query},
                expected_output={"eval_case_passed": True},
            )
            return getattr(item, "id", None)
        except Exception:
            return None
    except Exception:
        return None


def _link_dataset_run(client, case_id: str, trace_id: str, run_name: str) -> None:
    try:
        client.create_dataset_run_item(
            dataset_name=EVAL_DATASET_NAME,
            run_name=run_name,
            dataset_item_id=case_id,
            trace_id=trace_id,
        )
    except Exception:
        pass


def _create_score(client, trace_id: str, name: str, value: float, comment: str = "") -> None:
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment[:500] if comment else None,
        )
    except Exception:
        pass


def record_case_eval(
    trace_id: str | None,
    case: EvalCase,
    case_eval: CaseEvalResult,
    run: EvalRunResult | None = None,
    *,
    run_name: str | None = None,
) -> None:
    """Record evaluator pass/fail and metric scores on the Langfuse trace."""
    if not trace_id or not is_langfuse_enabled():
        return

    client = get_langfuse_client()
    if client is None:
        return

    run_name = run_name or os.getenv(
        "LANGFUSE_EVAL_RUN_NAME",
        f"offline_eval_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
    )

    _ensure_benchmark_dataset(client)
    _upsert_dataset_item(client, case)
    _link_dataset_run(client, case.id, trace_id, run_name)

    case_passed = all(item.passed for item in case_eval.evaluators)
    evaluators_payload = [item.model_dump() for item in case_eval.evaluators]

    try:
        client.update_trace(
            trace_id=trace_id,
            metadata={
                "eval_case_passed": case_passed,
                "evaluator_count": len(case_eval.evaluators),
                "failure_modes": case_eval.failure_modes,
            },
            output={
                "eval_case_passed": case_passed,
                "evaluators": evaluators_payload,
                "failure_modes": case_eval.failure_modes,
            },
        )
    except Exception:
        pass

    _create_score(
        client,
        trace_id,
        "eval_case_passed",
        1.0 if case_passed else 0.0,
        "; ".join(case_eval.failure_modes[:3]),
    )

    for item in case_eval.evaluators:
        _create_score(
            client,
            trace_id,
            _safe_score_name(item.name),
            1.0 if item.passed else 0.0,
            item.explanation,
        )
        if item.score is not None and item.name != "llm_judge":
            _create_score(
                client,
                trace_id,
                f"{_safe_score_name(item.name)}_score",
                float(item.score),
                item.explanation,
            )

        if item.name == "llm_judge":
            metrics = (item.details or {}).get("metrics") or {}
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and metric_data.get("score") is not None:
                    _create_score(
                        client,
                        trace_id,
                        f"llm_{metric_name}",
                        float(metric_data["score"]),
                        f"min={metric_data.get('min_score')} passed={metric_data.get('passed')}",
                    )

    if run is not None:
        _create_score(
            client,
            trace_id,
            "graph_success",
            1.0 if run.success else 0.0,
            run.error or "",
        )

    client.flush()
