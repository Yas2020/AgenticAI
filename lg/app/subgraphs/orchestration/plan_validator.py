from collections import deque, defaultdict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from app.core.state import MasterState

MAX_TASKS = 8
MAX_PLAN_ATTEMPTS = 3


def validate_dag(tasks):
    """
        Checks if the planner DAG is really acyclic! No cycle allowed!
    """

    graph = defaultdict(list)
    indegree = defaultdict(int)

    for task in tasks:
        for dep in task.depends_on:
            graph[dep].append(task.id)
            indegree[task.id] += 1

    queue = deque()

    for task in tasks:
        if indegree[task.id] == 0:
            queue.append(task.id)

    completed = 0

    while queue:
        node = queue.popleft()
        completed += 1

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return completed == len(tasks)


def missing_dep(tasks):
    """
        Checks if there is any missing dependencies in the planner DAG
    """
    task_ids = {t.id for t in tasks}

    for task in tasks:
        for dep in task.depends_on:
            if dep not in task_ids:
                return dep

    return None


def collect_plan_errors(tasks) -> list[str]:
    """Return structural DAG validation errors for a plan."""
    errors: list[str] = []
    if not tasks:
        errors.append("Plan is empty.")
        return errors

    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        errors.append("Task IDs are not unique.")

    if len(ids) > MAX_TASKS:
        errors.append(f"Too many tasks {len(ids)} in the plan! Max is {MAX_TASKS}.")

    if not validate_dag(tasks):
        errors.append("DAG contains circular dependencies.")

    dep = missing_dep(tasks)
    if dep:
        errors.append(f"DAG contains missing dependencies {dep}.")

    return errors


#  --- Node ---

def plan_validator(state: MasterState):

    errors = collect_plan_errors(state["plan"])

    if errors:
        attempt = state.get("plan_attempt_count", 0) + 1
        if attempt >= MAX_PLAN_ATTEMPTS:
            error_msg = (
                f"Could not produce a valid plan after {MAX_PLAN_ATTEMPTS} attempts.\n"
                f"Last errors:\n" + "\n".join(errors)
            )
            return {
                "messages": [AIMessage(content=error_msg)],
                "plan": [],
                "is_plan_valid": False,
                "plan_attempt_count": attempt,
            }

        error_msg = f"DAG is not valid:\n{'\n'.join(errors)}\n\n Please regenerate."
        return {
            "messages": [HumanMessage(content=error_msg)],
            "plan": [],
            "is_plan_valid": False,
            "plan_attempt_count": attempt,
        }

    return {"is_plan_valid": True, "plan_attempt_count": 0}


# ----- Condition -----

def route_valid_plan(state: MasterState):
    """Route based on the explicit state flag for plan validation."""

    if state["is_plan_valid"]:
        return "scheduler"

    if state.get("plan_attempt_count", 0) >= MAX_PLAN_ATTEMPTS:
        return END

    return "planning_architect"
