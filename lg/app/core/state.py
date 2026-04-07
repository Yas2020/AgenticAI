import operator
from typing import Annotated, List, Optional, Union
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from app.schemas.task import Task, TaskUpdate
from app.schemas.artifact import Artifact


class EvaluationRubric(BaseModel):
    """Handles scores."""
    score: int = Field(ge=0, le=10)
    critique: str
    is_hallucinated: bool
    missing_requirements: List[str]


# Reducer function: Tells LangGraph how to merge new tasks into the task in the global state
def update_plan_status(current_tasks: List[Task], updates: Union[List[Task], TaskUpdate, dict, List]) -> List[Task]:
    """
    This maps the updates by agents back to the plan 
    without overwriting the whole list.
    """
    
    # 1. Handle Initialization (If state is empty, the first batch is the full plan), or dictionary passed by LangGraph
    # 2. Case: Initial Plan Creation (current_tasks is empty)
    if not current_tasks:
        # Convert any dicts in updates to Task objects and return
        return [up if isinstance(up, Task) else Task(**up) for up in updates]
    
    if not updates:
        return current_tasks
    
    # 2. Ensure we are working with a list of updates (handle possible single objects returned by agents)
    if not isinstance(updates, list):
        updates = [updates]

    # 3. Normalize current state and updates: Convert raw dicts (from checkpoints) to Task objects
    # This fixes: Task object has no attribute id (when it was a dict)
    current_tasks = [t if isinstance(t, Task) else Task(**t) for t in current_tasks]
    # updates = [t if isinstance(t, TaskUpdate) else TaskUpdate(**t) for t in updates]

    
    # 4. Create map for existing tasks - remember current_tasks is a list of dictionaries- t's are of type dict
    task_map = {t.id: t for t in current_tasks}

    # This covers updates of type TaskUpdate from agent nodes or of type dict from the orchestrator
    for up in updates:
        is_dict = isinstance(up, dict)
        up_id = up["id"] if is_dict else up.id
        up_status = up["status"] if is_dict else up.status
        up_error_message = up["error_message"] if is_dict else up.error_message
        if up_id in task_map:
            # Update the specific task in the plan list
            task_map[up_id].status = up_status
            task_map[up_id].error_message = up_error_message
                
    return list(task_map.values())


    
class MasterState(MessagesState):
    """The source of truth for the entire Graph."""
    topic: str
    
    # Store the high-level plan of required tasks. LangGraph will now use 'update_plan_status' to merge agent returns
    plan: Annotated[List[Task], update_plan_status] = Field(default_factory=list)
    
    # Store research findings and quantitative outputs
    artifacts: Annotated[List[Artifact], operator.add] = Field(default_factory=list)
    
    is_plan_valid: bool = False
    is_query_valid: bool = False
    
    # Shared variables across subgraphs
    current_focus: Optional[str] = None
    is_complete: bool = False
    
    # Global Token/Cost monitoring
    total_cost: float = 0.0
    step_count: int = 0


class EvaluationState(MasterState):
    """Internal state for the Evaluator-Optimizer loop."""
    last_evaluation: Optional[EvaluationRubric] = None
    retry_count: int = 0
    max_retries: int = 3