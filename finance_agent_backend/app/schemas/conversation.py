from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"


class ConversationUpdate(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    id: str
    user_id: int
    title: str
    is_archived: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int = 0


class ChatMessageResponse(BaseModel):
    id: str
    conversation_id: str
    user_id: Optional[int] = None
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = []
    token_count: Optional[int] = None
    is_error: bool = False
    created_at: Optional[str] = None
