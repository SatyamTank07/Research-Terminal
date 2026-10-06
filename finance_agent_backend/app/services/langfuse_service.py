"""Langfuse Observability & Tracing Service.

Provides centralized integration with Langfuse following best practices:
- Framework callbacks for LangChain and LangGraph
- Session continuity (session_id mapped to conversation.id)
- User attribution (user_id mapped to user.id / username)
- Dynamic tags and metadata per query
- Graceful degradation if keys are missing
- Asynchronous buffer flushing
"""

from contextlib import contextmanager
import logging
import os
from typing import Any, Dict, Iterator, List, Optional
from app.config import settings

logger = logging.getLogger("finance_agent.services.langfuse")

_langfuse_client = None


def is_langfuse_configured() -> bool:
    """Checks whether Langfuse public and secret keys are configured."""
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


def get_langfuse_client():
    """Initializes and caches the Langfuse client if configured."""
    global _langfuse_client
    if not is_langfuse_configured():
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import get_client

        # Ensure environment variables are synchronized
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_BASE_URL"] = settings.LANGFUSE_BASE_URL
        os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

        _langfuse_client = get_client()
        logger.info("Langfuse client initialized successfully.")
        return _langfuse_client
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse client: {e}")
        return None


def create_langfuse_callback() -> Optional[Any]:
    """Creates a Langfuse CallbackHandler for LangChain and LangGraph."""
    if not is_langfuse_configured():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        # Ensure environment variables are synchronized
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_BASE_URL"] = settings.LANGFUSE_BASE_URL
        os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

        return CallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY)
    except Exception as e:
        logger.warning(f"Failed to create Langfuse CallbackHandler: {e}")
        return None


@contextmanager
def langfuse_trace_scope(
    trace_name: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    """Context manager setting up Langfuse trace attributes and yielding a CallbackHandler."""
    if not is_langfuse_configured():
        yield None
        return

    try:
        from langfuse import propagate_attributes

        # Ensure client is initialized
        get_langfuse_client()

        with propagate_attributes(
            trace_name=trace_name,
            session_id=str(session_id) if session_id else None,
            user_id=str(user_id) if user_id else None,
            tags=tags,
            metadata=metadata,
        ):
            callback = create_langfuse_callback()
            yield callback
    except Exception as e:
        logger.warning(f"Error in langfuse_trace_scope: {e}")
        yield None


def flush_langfuse() -> None:
    """Safely flushes any buffered Langfuse traces to the server."""
    if not is_langfuse_configured():
        return

    try:
        client = get_langfuse_client()
        if client and hasattr(client, "flush"):
            client.flush()
    except Exception as e:
        logger.debug(f"Error during Langfuse flush: {e}")
