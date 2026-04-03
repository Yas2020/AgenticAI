import asyncio
from datetime import datetime

from app.schemas.task import TaskUpdate
from app.core.state import MasterState
from app.schemas.artifact import Artifact

class AgentInput(MasterState):
    task_id: int
    
async def quant(state: AgentInput):
    """
    A generic mock node to test parallel execution and state merging.
    """
    # 1. Find the task assigned to this agent with specific task id
    # Your scheduler should have already set one to 'running'
    task = next(t for t in state["plan"] if t.id == state["task_id"])
    
    print(f"--- [MOCK] Executing specific task {task.id}: {task.description} ---")
    
    # 2. Simulate 'Work' (Network latency)
    await asyncio.sleep(1) 
    
    # 3. Create a Dummy Artifact
    mock_artifact = Artifact(
        artifact_type="quant", 
        task_id=task.id,
        source=task.agent,
        content=f"Mock data for {task.description}",
        timestamp=datetime.now().isoformat(),
        success=True,
        error=None
    )
    
    # 4. Return the 'Receipt' for the Reducer
    return {
        "artifacts": [mock_artifact],
        "plan": [TaskUpdate(id=task.id, status="completed", error_message=None)]
    }