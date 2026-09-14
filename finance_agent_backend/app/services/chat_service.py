import logging
import os
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.agents import AgentRegistry
from app.models import ChatMessage, Conversation, User
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("finance_agent.services.chat")


def process_chat(request: ChatRequest, user: User, db: Session) -> ChatResponse:
    """Orchestrates conversation lookup/creation, message history, agent execution, and persistence."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")

    # 1. Prepare conversation auto-title from first prompt snippet
    prompt_snippet = request.message.strip().replace("\n", " ")
    snippet_title = (
        (prompt_snippet[:38] + "...") if len(prompt_snippet) > 38 else (prompt_snippet or "New Chat")
    )

    # 2. Lookup or create conversation thread
    conversation = None
    if request.conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == request.conversation_id,
                Conversation.user_id == user.id,
            )
            .first()
        )

    if not conversation:
        conversation = Conversation(
            user_id=user.id,
            title=snippet_title,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    elif conversation.title in ("New Chat", "", None):
        conversation.title = snippet_title
        db.commit()

    # 3. Persist user message in PostgreSQL
    user_msg = ChatMessage(
        conversation_id=conversation.id,
        user_id=user.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 4. Load past messages for multi-turn conversational context (last 10 turns)
    past_messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.conversation_id == conversation.id,
            ChatMessage.id != user_msg.id,
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
        .all()
    )
    past_messages.reverse()

    history_payload = [
        {"role": msg.role, "content": msg.content}
        for msg in past_messages
        if msg.role in ("user", "assistant")
    ]
    history_payload.append({"role": "user", "content": request.message})

    # 5. Resolve and execute Agent via Registry (DIP / OCP / LSP)
    agent_type = request.agent_type or "financial_analyst"
    try:
        agent = AgentRegistry.get(agent_type)
        output = agent.run(history_payload)

        # 6. Persist assistant reply in PostgreSQL
        assistant_msg = ChatMessage(
            conversation_id=conversation.id,
            user_id=user.id,
            role="assistant",
            content=output.content,
            sources=output.sources,
            is_error=False,
        )
        db.add(assistant_msg)
        conversation.updated_at = func.now()
        db.commit()
        db.refresh(assistant_msg)

        return ChatResponse(
            response=output.content,
            conversation_id=conversation.id,
            message_id=assistant_msg.id,
            sources=output.sources,
        )
    except Exception as e:
        err_str = str(e)
        logger.error(f"Error during chat agent execution: {err_str}")
        try:
            err_msg = ChatMessage(
                conversation_id=conversation.id,
                user_id=user.id,
                role="assistant",
                content=f"Error communicating with agent: {err_str}",
                is_error=True,
            )
            db.add(err_msg)
            db.commit()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=err_str)
