# app/schemas/api.py

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class MessageInput(BaseModel):
    content: str


class ThreadConfig(BaseModel):
    configurable: Dict[str, Any]


class GraphRequest(BaseModel):
    messages: List[MessageInput]
    topic: str
    thread: Optional[ThreadConfig] = None