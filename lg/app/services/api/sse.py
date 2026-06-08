"""SSE helpers for graph streaming and HITL interrupt detection."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import Request
from langchain_core.load import dumpd
from langgraph.types import Command


def _iter_node_updates(chunk: Any) -> list[tuple[str, Any]]:
    if not isinstance(chunk, dict):
        return []
    payload = chunk.get("data", chunk)
    if not isinstance(payload, dict):
        return []
    return [(name, update) for name, update in payload.items() if name != "__interrupt__"]


def _format_update_event(node_name: str, state_update: Any) -> str:
    if (
        isinstance(state_update, dict)
        and state_update.get("messages")
    ):
        last_msg = state_update["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    else:
        content = dumpd(state_update)

    return json.dumps(
        {
            "event": "update",
            "node": node_name,
            "content": content,
            "state": dumpd(state_update),
        }
    )


def _interrupt_event(snapshot: Any, config: dict[str, Any]) -> str | None:
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if not interrupts:
        return None

    first = interrupts[0]
    message = first.value if hasattr(first, "value") else first
    if not isinstance(message, str):
        message = str(message)

    thread_id = config.get("configurable", {}).get("thread_id")
    return json.dumps(
        {
            "event": "interrupt",
            "message": message,
            "thread_id": thread_id,
        }
    )


async def stream_graph(
    graph,
    graph_input: dict[str, Any] | Command,
    config: dict[str, Any],
    request: Request,
) -> AsyncIterator[str]:
    try:
        async for chunk in graph.astream(
            graph_input,
            config=config,
            stream_mode="updates",
            version="v2",
        ):
            if await request.is_disconnected():
                break

            for node_name, state_update in _iter_node_updates(chunk):
                if state_update is None:
                    continue
                yield _format_update_event(node_name, state_update)

        snapshot = await graph.aget_state(config)
        interrupt_payload = _interrupt_event(snapshot, config)
        if interrupt_payload:
            yield interrupt_payload
            return

        yield json.dumps({"event": "done"})
    except Exception as exc:
        yield json.dumps({"event": "error", "message": str(exc)})
