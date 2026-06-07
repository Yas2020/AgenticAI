"""Unit tests for quant prompt construction (no LLM or MCP)."""

from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.artifact import Artifact
from app.schemas.task import Task
from app.subgraphs.quant_analyst.quant import _build_quant_messages

TASK = Task(
    id=2,
    agent="quant_analyst",
    description="Compute YoY cloud revenue growth.",
    depends_on=[1],
    status="running",
    error_message=None,
)


def _quant_state(**overrides) -> dict:
    base = {
        "topic": "equity research",
        "plan": [TASK],
        "artifacts": [],
        "messages": [],
        "retry_count": 0,
        "audit_feedback": None,
    }
    base.update(overrides)
    return base


def _failed_quant_artifact(**content) -> Artifact:
    return Artifact(
        artifact_type="quantitative_analyst",
        task_id=TASK.id,
        source="quant_analyst",
        content=content,
        success=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.mark.unit
def test_first_attempt_builds_system_and_task_messages():
    messages = _build_quant_messages(_quant_state(), TASK, "MSFT cloud: $26.1B")

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert "MSFT cloud: $26.1B" in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert TASK.description in messages[1].content
    assert "execute_quant_code" in messages[1].content


@pytest.mark.unit
def test_audit_retry_injects_feedback_and_previous_code():
    state = _quant_state(
        audit_feedback="FAIL: Revenue should be 26.1B, not 2.5B.",
        retry_count=1,
        artifacts=[
            _failed_quant_artifact(code="revenue = 2.5\nprint(revenue)"),
        ],
    )

    messages = _build_quant_messages(state, TASK, "research bundle")

    assert len(messages) == 3
    correction = messages[2].content
    assert "Auditor rejected" in correction
    assert "FAIL: Revenue should be 26.1B" in correction
    assert "revenue = 2.5" in correction
    assert "execute_quant_code" in correction


@pytest.mark.unit
def test_execution_retry_injects_stderr_from_last_artifact():
    state = _quant_state(
        retry_count=1,
        artifacts=[
            _failed_quant_artifact(stderr="NameError: name 'pd' is not defined"),
        ],
    )

    messages = _build_quant_messages(state, TASK, "research bundle")

    assert len(messages) == 3
    correction = messages[2].content
    assert "failed to execute" in correction
    assert "NameError: name 'pd' is not defined" in correction


@pytest.mark.unit
def test_audit_retry_takes_priority_over_execution_retry():
    """After an audit fail, quant should see auditor guidance—not only stderr."""
    state = _quant_state(
        audit_feedback="FAIL: wrong units.",
        retry_count=2,
        artifacts=[
            _failed_quant_artifact(
                code="bad = 1",
                stderr="RuntimeError: divide by zero",
            ),
        ],
    )

    messages = _build_quant_messages(state, TASK, "research bundle")

    assert "Auditor rejected" in messages[2].content
    assert "FAIL: wrong units." in messages[2].content
    assert "bad = 1" in messages[2].content
