from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db, get_single_user
from app.models import ChatMessage, Conversation
from app.schemas.conversation import (
    ChatMessageResponse,
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db)):
    """Returns all conversations for the authorized user sorted by most recent activity."""
    user = get_single_user(db)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id, Conversation.is_archived == False)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [conv.to_dict() for conv in conversations]


@router.post("", response_model=ConversationResponse)
def create_conversation(req: ConversationCreate, db: Session = Depends(get_db)):
    """Creates a new empty conversation session."""
    user = get_single_user(db)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    conv = Conversation(
        user_id=user.id,
        title=req.title or "New Chat",
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv.to_dict()


@router.get("/{conversation_id}/messages", response_model=List[ChatMessageResponse])
def get_conversation_messages(conversation_id: str, db: Session = Depends(get_db)):
    """Retrieves all messages for a specific conversation in chronological order."""
    user = get_single_user(db)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [m.to_dict() for m in messages]


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: str, req: ConversationUpdate, db: Session = Depends(get_db)
):
    """Renames an existing conversation."""
    user = get_single_user(db)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if req.title and req.title.strip():
        conv.title = req.title.strip()
        db.commit()
        db.refresh(conv)
    return conv.to_dict()


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Deletes a conversation and all its associated messages."""
    user = get_single_user(db)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db.delete(conv)
    db.commit()
    return {"status": "deleted", "id": conversation_id}


@router.delete("/{conversation_id}/messages")
def clear_conversation_messages(conversation_id: str, db: Session = Depends(get_db)):
    """Clears all messages within a conversation without deleting the thread."""
    user = get_single_user(db)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation_id).delete()
    conv.updated_at = func.now()
    db.commit()
    return {"status": "cleared", "conversation_id": conversation_id}
