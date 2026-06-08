"""Persist lightweight online user feedback (file + optional Langfuse score)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.services.observability.langfuse_tracer import get_langfuse_client, is_langfuse_enabled

Rating = Literal["up", "down"]


def _feedback_dir() -> Path:
    root = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    path = root / "feedback"
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_user_feedback(
    *,
    thread_id: str,
    rating: Rating,
    comment: str | None = None,
    trace_id: str | None = None,
) -> dict:
    entry = {
        "thread_id": thread_id,
        "rating": rating,
        "score": 1.0 if rating == "up" else 0.0,
        "comment": comment,
        "trace_id": trace_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    log_path = _feedback_dir() / "feedback.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")

    langfuse_recorded = False
    if is_langfuse_enabled():
        client = get_langfuse_client()
        if client is not None:
            try:
                client.create_score(
                    trace_id=trace_id or thread_id,
                    name="user_feedback",
                    value=entry["score"],
                    comment=comment or f"thread={thread_id} rating={rating}",
                )
                client.flush()
                langfuse_recorded = True
            except Exception:
                pass

    return {
        "status": "recorded",
        "path": str(log_path),
        "langfuse_recorded": langfuse_recorded,
    }
