import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.agents import AgentRegistry
from app.models import ChatMessage, Conversation, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.langfuse_service import flush_langfuse, langfuse_trace_scope

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


def _prepare_chat_turn(
    request: ChatRequest, user: User, db: Session
) -> Tuple[Conversation, ChatMessage, List[Dict[str, str]]]:
    """Resolves conversation, persists incoming user message, and constructs history payload."""
    conversation = _resolve_conversation(request=request, user=user, db=db)

    user_msg = ChatMessage(
        conversation_id=conversation.id,
        user_id=user.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

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

    return conversation, user_msg, history_payload


def _finalize_assistant_turn(
    conversation: Conversation,
    user: User,
    content: str,
    sources: Optional[List[Dict[str, Any]]],
    updated_session_state: Optional[Dict[str, Any]],
    db: Session,
    is_error: bool = False,
) -> ChatMessage:
    """Persists assistant reply, updates conversation session state and timestamp."""
    assistant_msg = ChatMessage(
        conversation_id=conversation.id,
        user_id=user.id,
        role="assistant",
        content=content,
        sources=sources or [],
        is_error=is_error,
    )
    db.add(assistant_msg)
    if updated_session_state is not None:
        conversation.session_state = dict(updated_session_state)
        flag_modified(conversation, "session_state")
    conversation.updated_at = func.now()
    db.commit()
    db.refresh(assistant_msg)
    return assistant_msg


def process_chat(request: ChatRequest, user: User, db: Session) -> ChatResponse:
    """Orchestrates conversation lookup/creation, message history, agent execution, and persistence."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")

    conversation, user_msg, history_payload = _prepare_chat_turn(request=request, user=user, db=db)

    agent_type = request.agent_type or "multi_agent"
    trace_name = f"finance-agent:{agent_type}"
    tags = ["finance-agent", agent_type]
    metadata = {
        "conversation_id": str(conversation.id),
        "user_id": str(user.id),
        "username": getattr(user, "username", "finance_user"),
        "agent_type": agent_type,
    }

    try:
        with langfuse_trace_scope(
            trace_name=trace_name,
            session_id=str(conversation.id),
            user_id=str(user.id),
            tags=tags,
            metadata=metadata,
        ) as callback:
            callbacks = [callback] if callback else None
            agent = AgentRegistry.get(agent_type)
            try:
                output = agent.run(
                    history_payload,
                    session_state=conversation.session_state,
                    callbacks=callbacks,
                )
            except TypeError:
                try:
                    output = agent.run(history_payload, session_state=conversation.session_state)
                except TypeError:
                    output = agent.run(history_payload)

            assistant_msg = _finalize_assistant_turn(
                conversation=conversation,
                user=user,
                content=output.content,
                sources=output.sources,
                updated_session_state=output.updated_session_state,
                db=db,
            )

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
            _finalize_assistant_turn(
                conversation=conversation,
                user=user,
                content=f"Error communicating with agent: {err_str}",
                sources=[],
                updated_session_state=None,
                db=db,
                is_error=True,
            )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=err_str)
    finally:
        flush_langfuse()


async def stream_chat_service(
    request: ChatRequest, user: User, db: Session
) -> AsyncIterator[str]:
    """Streams real-time intermediate node progress milestones, token deltas, and final response via SSE."""
    if not os.getenv("OPENAI_API_KEY"):
        yield f"data: {json.dumps({'type': 'error', 'message': 'OPENAI_API_KEY is not set.'})}\n\n"
        return

    conversation, user_msg, history_payload = _prepare_chat_turn(request=request, user=user, db=db)

    agent_type = request.agent_type or "multi_agent"
    trace_name = f"finance-agent:stream:{agent_type}"
    tags = ["finance-agent", "streaming", agent_type]
    metadata = {
        "conversation_id": str(conversation.id),
        "user_id": str(user.id),
        "username": getattr(user, "username", "finance_user"),
        "agent_type": agent_type,
    }

    try:
        with langfuse_trace_scope(
            trace_name=trace_name,
            session_id=str(conversation.id),
            user_id=str(user.id),
            tags=tags,
            metadata=metadata,
        ) as callback:
            callbacks = [callback] if callback else None
            agent = AgentRegistry.get(agent_type)
            if hasattr(agent, "astream_run"):
                async for event in agent.astream_run(
                    user_query=request.message,
                    messages=history_payload,
                    session_state=conversation.session_state,
                    callbacks=callbacks,
                ):
                    event_type = event.get("type")
                    if event_type == "result":
                        response_content = event.get("response", "")
                        sources = event.get("sources", [])
                        updated_state = event.get("updated_session_state")

                        # If tokens were not streamed natively during node execution,
                        # stream response in word chunks to ensure token-by-token UI typing
                        if not event.get("tokens_streamed", False) and response_content:
                            words = response_content.split(" ")
                            chunk_size = 6
                            for i in range(0, len(words), chunk_size):
                                chunk_words = words[i : i + chunk_size]
                                chunk_text = " ".join(chunk_words)
                                if i + chunk_size < len(words):
                                    chunk_text += " "
                                yield f"data: {json.dumps({'type': 'token', 'delta': chunk_text})}\n\n"
                                await asyncio.sleep(0.005)

                        assistant_msg = _finalize_assistant_turn(
                            conversation=conversation,
                            user=user,
                            content=response_content,
                            sources=sources,
                            updated_session_state=updated_state,
                            db=db,
                        )

                        payload = {
                            "type": "result",
                            "response": response_content,
                            "conversation_id": conversation.id,
                            "message_id": assistant_msg.id,
                            "sources": sources,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif event_type == "token":
                        yield f"data: {json.dumps(event)}\n\n"
                    else:
                        yield f"data: {json.dumps(event)}\n\n"
            else:
                # Fallback for non-streaming agents
                try:
                    output = await asyncio.to_thread(agent.run, history_payload, conversation.session_state, callbacks)
                except TypeError:
                    try:
                        output = await asyncio.to_thread(agent.run, history_payload, conversation.session_state)
                    except TypeError:
                        output = await asyncio.to_thread(agent.run, history_payload)

                if output.content:
                    words = output.content.split(" ")
                    chunk_size = 6
                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i : i + chunk_size]
                        chunk_text = " ".join(chunk_words)
                        if i + chunk_size < len(words):
                            chunk_text += " "
                        yield f"data: {json.dumps({'type': 'token', 'delta': chunk_text})}\n\n"
                        await asyncio.sleep(0.005)

                assistant_msg = _finalize_assistant_turn(
                    conversation=conversation,
                    user=user,
                    content=output.content,
                    sources=output.sources,
                    updated_session_state=output.updated_session_state,
                    db=db,
                )

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
            _finalize_assistant_turn(
                conversation=conversation,
                user=user,
                content=f"Error communicating with agent: {err_str}",
                sources=[],
                updated_session_state=None,
                db=db,
                is_error=True,
            )
        except Exception:
            pass
        yield f"data: {json.dumps({'type': 'error', 'message': err_str})}\n\n"
    finally:
        flush_langfuse()
