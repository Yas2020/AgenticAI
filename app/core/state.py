from typing import Annotated, List, Optional, Union
from app.schemas.task import Task, TaskUpdate
from app.schemas.artifact import Artifact
from pydantic import BaseModel, Field
import operator
from langgraph.graph import MessagesState


class EvaluationRubric(BaseModel):
    """Handles scores."""
    score: int = Field(ge=0, le=10)
    critique: str
    is_hallucinated: bool
    missing_requirements: List[str]


# Reducer function: Tells LangGraph how to merge new data into the state
def update_plan_status(current_tasks: List[Task], updates: Union[List[Task], TaskUpdate, dict, List]) -> List[Task]:
    """
    This maps the updates by agents back to the plan 
    without overwriting the whole list.
    """
    
    # 1. Handle Initialization (If state is empty, the first batch is the full plan), or dictionary passed by LangGraph
    if not current_tasks:
        # Ensure every item is a Task object, not a raw dict
        return [t if isinstance(t, Task) else Task(**t) for t in updates]

    # 3. Ensure we are working with a list of updates (handle possible single objects returned by agents)
    if not isinstance(updates, list):
        updates = [updates]

    # 3. Create map for existing tasks
    task_map = {t.id: t for t in current_tasks}

    for up in updates:
        # Get ID and Status safely whether 'up' is a Pydantic object or a Dict
        is_dict = isinstance(up, dict)
        up_id = up["id"] if is_dict else up.id
        up_status = up["status"] if is_dict else up.status
        
        if up_id in task_map:
            # Update the specific task in the master list
            task_map[up_id].status = up_status
            
            # Handle optional error messages
            err = up.get("error_message") if is_dict else getattr(up, "error_message", None)
            if err:
                task_map[up_id].error_message = err
                
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