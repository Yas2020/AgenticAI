"""Run a single evaluation case through the graph."""

import os
import time
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from app.schemas.task import Task
from app.services.observability.llm_pricing import estimate_cost_usd
from evals.test_cases.schemas import EvalRunResult, TokenUsage


class EvalRunCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tool_call_count = 0
        self.token_usage = TokenUsage()

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.tool_call_count += 1

    def _add_usage(self, usage: Any) -> None:
        if not usage:
            return
        if isinstance(usage, dict):
            inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            total = usage.get("total_tokens") or (inp + out)
        else:
            inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
            total = getattr(usage, "total_tokens", None) or (inp + out)

        self.token_usage.input_tokens += int(inp)
        self.token_usage.output_tokens += int(out)
        self.token_usage.total_tokens += int(total)

    def on_llm_end(self, response, **kwargs) -> None:
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            self._add_usage(llm_output.get("token_usage"))
            self._add_usage(llm_output.get("usage"))

        for generation_list in response.generations:
            for generation in generation_list:
                message = generation.message
                self._add_usage(getattr(message, "usage_metadata", None))
                response_metadata = getattr(message, "response_metadata", None) or {}
                if isinstance(response_metadata, dict):
                    self._add_usage(response_metadata.get("token_usage"))


def _serialize_artifacts(artifacts: list[Any]) -> list[dict[str, Any]]:
    serialized = []
    for artifact in artifacts or []:
        if hasattr(artifact, "model_dump"):
            serialized.append(artifact.model_dump())
        elif isinstance(artifact, dict):
            serialized.append(artifact)
    return serialized


def _extract_final_response(artifacts: list[dict[str, Any]], messages: list[Any]) -> str | None:
    for artifact in reversed(artifacts):
        if artifact.get("artifact_type") == "final_report":
            content = artifact.get("content")
            return content if isinstance(content, str) else str(content)
    for message in reversed(messages or []):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return None


async def run_case(graph, case, config: dict[str, Any] | None = None) -> EvalRunResult:
    config = dict(config or {})
    config.setdefault("recursion_limit", int(os.getenv("EVAL_GRAPH_RECURSION_LIMIT", "50")))
    configurable = config.setdefault("configurable", {})
    configurable.setdefault("thread_id", f"eval-{case.id}")

    callbacks = list(config.get("callbacks", []))
    run_callback = EvalRunCallback()
    callbacks.append(run_callback)
    config["callbacks"] = callbacks

    inputs = {
        "messages": [HumanMessage(content=case.input_query)],
        "topic": case.topic,
    }

    execution_path: list[str] = []
    start = time.perf_counter()
    error = None
    final_values: dict[str, Any] = {}

    try:
        async for chunk in graph.astream(inputs, config=config, stream_mode="updates"):
            for node_name in chunk:
                if node_name not in execution_path:
                    execution_path.append(node_name)

        snapshot = await graph.aget_state(config)
        final_values = snapshot.values if snapshot else {}
    except Exception as exc:
        error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    artifacts = _serialize_artifacts(final_values.get("artifacts", []))
    plan_raw = final_values.get("plan", [])
    plan = [t if isinstance(t, Task) else Task(**t) for t in plan_raw]

    return EvalRunResult(
        case_id=case.id,
        category=case.category,
        success=error is None,
        latency_ms=round(latency_ms, 2),
        execution_path=execution_path,
        final_state={
            "is_query_valid": final_values.get("is_query_valid"),
            "is_plan_valid": final_values.get("is_plan_valid"),
            "step_count": final_values.get("step_count", 0),
        },
        artifacts=artifacts,
        plan=plan,
        is_query_valid=final_values.get("is_query_valid"),
        is_plan_valid=final_values.get("is_plan_valid"),
        tool_call_count=run_callback.tool_call_count,
        token_usage=run_callback.token_usage,
        estimated_cost_usd=estimate_cost_usd(
            input_tokens=run_callback.token_usage.input_tokens,
            output_tokens=run_callback.token_usage.output_tokens,
        ),
        error=error,
        final_response=_extract_final_response(
            artifacts, final_values.get("messages", [])
        ),
    )
