# app/schemas/api.py

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MessageInput(BaseModel):
    content: str


class ThreadConfig(BaseModel):
    configurable: Dict[str, Any]


class GraphRequest(BaseModel):
    messages: List[MessageInput]
    topic: str
    thread: Optional[ThreadConfig] = None


class GraphResumeRequest(BaseModel):
    resume: str = Field(min_length=1, description="Clarified user query after HITL interrupt.")
    thread: ThreadConfig


class FeedbackRequest(BaseModel):
    thread_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None
    trace_id: Optional[str] = None