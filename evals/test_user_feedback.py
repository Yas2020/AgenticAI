"""Unit tests for online user feedback recording."""

import json

import pytest

from app.services.observability import user_feedback


@pytest.mark.unit
def test_record_user_feedback_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(user_feedback, "is_langfuse_enabled", lambda: False)

    result = user_feedback.record_user_feedback(
        thread_id="demo-thread",
        rating="up",
        comment="Useful report",
    )

    log_path = tmp_path / "feedback" / "feedback.jsonl"
    assert log_path.exists()
    entry = json.loads(log_path.read_text().splitlines()[-1])
    assert entry["thread_id"] == "demo-thread"
    assert entry["rating"] == "up"
    assert entry["score"] == 1.0
    assert result["status"] == "recorded"
