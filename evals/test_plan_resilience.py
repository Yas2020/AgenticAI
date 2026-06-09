"""Unit tests for bounded plan regeneration and routing."""

import pytest
from langgraph.graph import END

from app.schemas.task import Task
from app.subgraphs.orchestration.plan_validator import (
    MAX_PLAN_ATTEMPTS,
    plan_validator,
    route_valid_plan,
)


@pytest.mark.unit
def test_plan_validator_increments_attempt_count_on_invalid_dag():
    cyclic_plan = [
        Task(id=1, agent="research", description="a", depends_on=[2]),
        Task(id=2, agent="quant_analyst", description="b", depends_on=[1]),
    ]
    state = {"plan": cyclic_plan, "plan_attempt_count": 0}

    result = plan_validator(state)

    assert result["is_plan_valid"] is False
    assert result["plan_attempt_count"] == 1
    assert "circular" in result["messages"][0].content.lower()


@pytest.mark.unit
def test_plan_validator_resets_attempt_count_on_success():
    valid_plan = [
        Task(id=1, agent="research", description="search", depends_on=[]),
        Task(id=2, agent="analyst", description="report", depends_on=[1]),
    ]
    state = {"plan": valid_plan, "plan_attempt_count": 2}

    result = plan_validator(state)

    assert result["is_plan_valid"] is True
    assert result["plan_attempt_count"] == 0


@pytest.mark.unit
def test_plan_validator_final_message_at_max_attempts():
    state = {"plan": [], "plan_attempt_count": MAX_PLAN_ATTEMPTS - 1}

    result = plan_validator(state)

    assert result["plan_attempt_count"] == MAX_PLAN_ATTEMPTS
    assert str(MAX_PLAN_ATTEMPTS) in result["messages"][0].content


@pytest.mark.unit
def test_route_valid_plan_ends_after_max_attempts():
    state = {"is_plan_valid": False, "plan_attempt_count": MAX_PLAN_ATTEMPTS}

    assert route_valid_plan(state) is END


@pytest.mark.unit
def test_route_valid_plan_retries_before_max_attempts():
    state = {"is_plan_valid": False, "plan_attempt_count": 1}

    assert route_valid_plan(state) == "planning_architect"
