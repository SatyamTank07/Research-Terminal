from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db, get_single_user
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import process_chat, stream_chat_service

router = APIRouter(tags=["Chat"])


@router.post("/chat", responses={200: {"model": ChatResponse, "description": "Chat response JSON or text/event-stream"}})
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Unified chat endpoint supporting both real-time SSE streaming (stream=True) and synchronous JSON (stream=False)."""
    # Enforce single user restriction
    user = get_single_user(db)
    if not user:
        raise HTTPException(
            status_code=403, detail="Unauthorized: No active single user profile configured."
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is deactivated.")

    if request.stream:
        return StreamingResponse(
            stream_chat_service(request=request, user=user, db=db),
            media_type="text/event-stream",
        )

    return process_chat(request=request, user=user, db=db)
