"""Langfuse tracing helpers for graph runs and offline evals.

One trace per eval case: pre-generate trace_id and pass it to CallbackHandler
via trace_context so all LangGraph steps nest under a single trace.

Per-generation costs are recorded by the LangChain callback (Langfuse v4 OTEL).
``trace.end()`` also writes an ``estimated_cost_usd`` score from eval token totals.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from langfuse import get_client
from langfuse.langchain import CallbackHandler

_LANGFUSE_INITIALIZED = False


def is_langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def ensure_langfuse_initialized() -> None:
    """Initialize the Langfuse singleton once with Cloud v4 OTEL ingestion headers."""
    global _LANGFUSE_INITIALIZED
    if _LANGFUSE_INITIALIZED or not is_langfuse_enabled():
        return

    ingestion_version = os.getenv("LANGFUSE_OTEL_INGESTION_VERSION", "4")
    additional_headers = {"x-langfuse-ingestion-version": ingestion_version}

    try:
        from langfuse import Langfuse

        Langfuse(additional_headers=additional_headers)
    except TypeError:
        # Older SDKs without additional_headers support.
        get_client()
    except Exception:
        pass

    _LANGFUSE_INITIALIZED = True


def get_langfuse_client():
    if not is_langfuse_enabled():
        return None
    ensure_langfuse_initialized()
    return get_client()


def _new_trace_id(client) -> str:
    if hasattr(client, "create_trace_id"):
        return client.create_trace_id()
    return uuid.uuid4().hex


@dataclass
class EvalTraceContext:
    case_id: str
    category: str
    trace_id: str | None = None
    handler: CallbackHandler | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> "EvalTraceContext":
        if not is_langfuse_enabled():
            return self

        client = get_langfuse_client()
        if client is None:
            return self

        self.trace_id = _new_trace_id(client)
        tags = ["offline_eval", self.category, self.case_id]

        try:
            client.update_trace(
                trace_id=self.trace_id,
                name=f"offline_eval:{self.case_id}",
                session_id="offline_eval",
                tags=tags,
                metadata={"case_id": self.case_id, "category": self.category, **self.metadata},
                input={"case_id": self.case_id, "category": self.category},
            )
        except Exception:
            pass

        trace_context = {"trace_id": self.trace_id}
        try:
            self.handler = CallbackHandler(
                trace_context=trace_context,
                update_trace=True,
            )
        except TypeError:
            self.handler = CallbackHandler(trace_context=trace_context)
        except Exception:
            self.handler = None

        return self

    def run_config(self, base: dict[str, Any] | None = None) -> dict[str, Any]:
        config = dict(base or {})
        if self.handler is None:
            return config

        config["run_name"] = f"offline_eval:{self.case_id}"
        config["callbacks"] = [self.handler]

        metadata = dict(config.get("metadata", {}))
        metadata.update(
            {
                "langfuse_session_id": "offline_eval",
                "langfuse_tags": ["offline_eval", self.category, self.case_id],
                "langfuse_trace_id": self.trace_id,
                "case_id": self.case_id,
                "category": self.category,
                **self.metadata,
            }
        )
        config["metadata"] = metadata
        return config

    def end(self, output: dict[str, Any] | None = None) -> None:
        output = output or {}
        client = get_langfuse_client()
        if client is None:
            return

        trace_id = self.trace_id or getattr(self.handler, "last_trace_id", None)
        if not trace_id:
            client.flush()
            return

        token_usage = output.get("token_usage") or {}
        estimated_cost_usd = output.get("estimated_cost_usd")
        trace_output = {
            "passed": output.get("passed"),
            "execution_path": output.get("execution_path"),
            "final_response": _truncate(output.get("final_response")),
            "latency_ms": output.get("latency_ms"),
            "tool_call_count": output.get("tool_call_count"),
        }
        trace_metadata = {
            "case_id": self.case_id,
            "category": self.category,
            "estimated_cost_usd": estimated_cost_usd,
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }
        tags = ["offline_eval", self.category, self.case_id]

        try:
            client.update_trace(
                trace_id=trace_id,
                name=f"offline_eval:{self.case_id}",
                tags=tags,
                metadata=trace_metadata,
                output=trace_output,
            )
        except Exception:
            pass

        try:
            client.create_score(
                trace_id=trace_id,
                name="estimated_cost_usd",
                value=float(estimated_cost_usd or 0),
                comment=(
                    f"tokens in={trace_metadata['input_tokens']} "
                    f"out={trace_metadata['output_tokens']}"
                ),
            )
        except Exception:
            pass

        client.flush()


def _truncate(value: Any, limit: int = 4000) -> Any:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def create_eval_trace(case_id: str, category: str, **metadata: Any) -> EvalTraceContext:
    ensure_langfuse_initialized()
    return EvalTraceContext(
        case_id=case_id,
        category=category,
        metadata=metadata,
    ).start()


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is not None:
        client.flush()
