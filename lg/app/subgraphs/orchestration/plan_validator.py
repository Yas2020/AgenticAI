from collections import deque, defaultdict
from langchain_core.messages import HumanMessage

from app.core.state import MasterState


MAX_TASKS = 8

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


#  --- Node ---

def plan_validator(state: MasterState):
    
    ids = [t.id for t in state["plan"]]
    errors = []

    # Check Ids uniqueness
    if len(ids) != len(set(ids)):
        errors.append("Task IDs are not unique.")
    
    # Check tasks limit
    if len(ids) > MAX_TASKS:
        errors.append(f"Too many tasks {len(ids)} in the plan! Max is {MAX_TASKS}.")

    # Check possible cycles in dependencies
    if not validate_dag(state["plan"]):
        errors.append("DAG contains circular dependencies.")
        
    # Check for missing dependencies
    dep = missing_dep(state["plan"])
    if dep:
        errors.append(f"DAG contains missing dependencies {dep}.")
        
    if errors:
        error_msg = f"DAG is not valid:\n{'\n'.join(errors)}\n\n Please regenerate."
        # CRITICAL: We return the error message AND clear the plan
        return {
            "messages": [HumanMessage(content=error_msg)],
            "plan": [], # This triggers the Reducer to 'reset' the plan
            "is_plan_valid": False 
        }
    
    # If valid, we set the flag to True
    return {"is_plan_valid": True}  
        
  
# ----- Condition ---
  
def route_valid_plan(state: MasterState):
    """Route based on the explicit state flag for plan validation"""
    
    if state["is_plan_valid"]:
        return "scheduler"
    
    return "planning_architect"