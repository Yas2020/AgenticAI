from langgraph.graph import END
from langgraph.types import Send
from app.core.state import MasterState


def _task_map(state: MasterState) -> dict[int, object]:
    return {t.id: t for t in state["plan"]}


def _promote_pending_tasks(state: MasterState) -> list[dict]:
    """Mark tasks ready or failed (skipped) based on dependency outcomes."""
    task_map = _task_map(state)
    updates: list[dict] = []

    for task in state["plan"]:
        if task.status != "pending":
            continue
        if not task.depends_on:
            task.status = "ready"
            continue

        dep_statuses = [
            task_map[dep_id].status
            for dep_id in task.depends_on
            if dep_id in task_map
        ]
        if not dep_statuses or len(dep_statuses) != len(task.depends_on):
            updates.append(
                {
                    "id": task.id,
                    "status": "failed",
                    "error_message": "Invalid or missing dependency in plan.",
                }
            )
            continue

        if any(status == "failed" for status in dep_statuses):
            if task.agent == "analyst" and any(status == "completed" for status in dep_statuses):
                task.status = "ready"
            else:
                failed_deps = [
                    dep_id
                    for dep_id in task.depends_on
                    if dep_id in task_map and task_map[dep_id].status == "failed"
                ]
                updates.append(
                    {
                        "id": task.id,
                        "status": "failed",
                        "error_message": (
                            f"Skipped: upstream task(s) {failed_deps} failed."
                        ),
                    }
                )
            continue

        if all(status == "completed" for status in dep_statuses):
            task.status = "ready"

    return updates


def scheduler(state: MasterState):
    """
    Scheduler: promote pending tasks when dependencies are satisfied, skip or
    partially run analyst when upstream failures occur, and dispatch ready work.
    """
    if all(t.status in ("completed", "failed") for t in state["plan"]):
        return {}

    skip_updates = _promote_pending_tasks(state)

    running_updates = [
        {"id": t.id, "status": "running", "error_message": None}
        for t in state["plan"]
        if t.status == "ready"
    ]

    if not skip_updates and not running_updates:
        return {}

    return {"plan": skip_updates + running_updates}


def route_to_agents(state: MasterState):
    if all(t.status in ("completed", "failed") for t in state["plan"]):
        return END

    return [
        Send(
            t.agent,
            {
                **state,
                "task_id": t.id,
            },
        )
        for t in state["plan"]
        if t.status == "running"
    ]
