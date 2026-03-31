from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class Task(BaseModel):
    """A granular step in the investment research process."""
    id: int
    agent: Literal["research", "analyst", "quant_sandbox", "vector_db"]
    description: str
    depends_on: List[int] = Field(default_factory=list, description="IDs of tasks that must complete before this task starts")
    status: Literal["pending", "ready", "running", "completed", "failed"] = "pending"
    error_message: Optional[str] = None
    retries: int = 0 

# Partial Task update 
class TaskUpdate(BaseModel):
    id: int
    status: Literal["completed", "failed"]
    error_message: Optional[str] = None

