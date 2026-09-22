import asyncio
import json
import logging
import os
from typing import AsyncIterator
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.agents import AgentRegistry
from app.models import ChatMessage, Conversation, User
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("finance_agent.services.chat")


def _resolve_conversation(request: ChatRequest, user: User, db: Session) -> Conversation:
    """Helper to retrieve or initialize a conversation thread."""
    prompt_snippet = request.message.strip().replace("\n", " ")
    snippet_title = (
        (prompt_snippet[:38] + "...") if len(prompt_snippet) > 38 else (prompt_snippet or "New Chat")
    )

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

    return conversation


def process_chat(request: ChatRequest, user: User, db: Session) -> ChatResponse:
    """Orchestrates conversation lookup/creation, message history, multi-agent execution, and persistence."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")

    # 1. Resolve conversation thread
    conversation = _resolve_conversation(request=request, user=user, db=db)

    # 2. Persist user message in PostgreSQL
    user_msg = ChatMessage(
        conversation_id=conversation.id,
        user_id=user.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 3. Load past messages for multi-turn conversational context (last 10 turns)
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

    # 4. Resolve and execute Agent via Registry (defaults to multi-agent system)
    agent_type = request.agent_type or "multi_agent"
    if agent_type == "financial_analyst":
        agent_type = "multi_agent"

    try:
        agent = AgentRegistry.get(agent_type)
        try:
            output = agent.run(history_payload, session_state=conversation.session_state)
        except TypeError:
            output = agent.run(history_payload)

        # Update session_state if provided by agent
        if output.updated_session_state is not None:
            conversation.session_state = dict(output.updated_session_state)
            flag_modified(conversation, "session_state")

        # 5. Persist assistant reply in PostgreSQL
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


async def stream_chat_service(
    request: ChatRequest, user: User, db: Session
) -> AsyncIterator[str]:
    """Streams real-time intermediate node progress milestones and final response via SSE."""
    if not os.getenv("OPENAI_API_KEY"):
        yield f"data: {json.dumps({'type': 'error', 'message': 'OPENAI_API_KEY is not set.'})}\n\n"
        return

    # 1. Resolve conversation thread
    conversation = _resolve_conversation(request=request, user=user, db=db)

    # 2. Persist user message in PostgreSQL
    user_msg = ChatMessage(
        conversation_id=conversation.id,
        user_id=user.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 3. Load past messages for multi-turn conversational context (last 10 turns)
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

    # 4. Resolve agent (defaulting to multi_agent)
    agent_type = request.agent_type or "multi_agent"
    if agent_type == "financial_analyst":
        agent_type = "multi_agent"

    try:
        agent = AgentRegistry.get(agent_type)
        if hasattr(agent, "astream_run"):
            async for event in agent.astream_run(
                user_query=request.message,
                messages=history_payload,
                session_state=conversation.session_state,
            ):
                if event.get("type") == "result":
                    # Persist assistant reply in PostgreSQL
                    assistant_msg = ChatMessage(
                        conversation_id=conversation.id,
                        user_id=user.id,
                        role="assistant",
                        content=event.get("response", ""),
                        sources=event.get("sources", []),
                        is_error=False,
                    )
                    db.add(assistant_msg)
                    if event.get("updated_session_state") is not None:
                        conversation.session_state = dict(event["updated_session_state"])
                        flag_modified(conversation, "session_state")
                    conversation.updated_at = func.now()
                    db.commit()
                    db.refresh(assistant_msg)

                    payload = {
                        "type": "result",
                        "response": event.get("response", ""),
                        "conversation_id": conversation.id,
                        "message_id": assistant_msg.id,
                        "sources": event.get("sources", []),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        else:
            # Fallback for non-streaming agents
            try:
                output = await asyncio.to_thread(agent.run, history_payload, conversation.session_state)
            except TypeError:
                output = await asyncio.to_thread(agent.run, history_payload)
            assistant_msg = ChatMessage(
                conversation_id=conversation.id,
                user_id=user.id,
                role="assistant",
                content=output.content,
                sources=output.sources,
                is_error=False,
            )
            db.add(assistant_msg)
            if output.updated_session_state is not None:
                conversation.session_state = dict(output.updated_session_state)
                flag_modified(conversation, "session_state")
            conversation.updated_at = func.now()
            db.commit()
            db.refresh(assistant_msg)

            payload = {
                "type": "result",
                "response": output.content,
                "conversation_id": conversation.id,
                "message_id": assistant_msg.id,
                "sources": output.sources,
            }
            yield f"data: {json.dumps(payload)}\n\n"

    except Exception as e:
        err_str = str(e)
        logger.error(f"Error during stream chat execution: {err_str}")
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
        yield f"data: {json.dumps({'type': 'error', 'message': err_str})}\n\n"
