from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    agent_type: Optional[str] = "financial_analyst"


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    sources: List[Dict[str, Any]] = []
