"""Conversational Equity Research & Financial AI Assistant.

Handles multi-turn conversational dialogue, capabilities overview, financial
concepts Q&A, and follow-up inquiries without requiring rigid 10-K filing lookup.
All prompts are dynamically loaded from prompt_conversational.j2.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.base import AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.database import SessionLocal
from app.models import Document

logger = logging.getLogger("finance_agent.agents.conversational")


@AgentRegistry.register("conversational_analyst")
@AgentRegistry.register("conversational")
class ConversationalAnalystAgent(BaseAgent):
    """Institutional equity research assistant specialized in natural conversational interaction."""

    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        self.model_name = model_name
        self._cached_llm = None

    def _get_llm(self):
        if self._cached_llm is not None:
            return self._cached_llm

        load_dotenv(override=True)
        model_clean = self.model_name.replace("openai:", "")
        llm = ChatOpenAI(
            model=model_clean,
            temperature=0.3,
            max_retries=3,
        )
        self._cached_llm = llm
        return llm

    def _get_catalog_summary(self) -> str:
        """Retrieves summary of ingested company 10-K filings from database."""
        db = SessionLocal()
        try:
            available_docs = (
                db.query(Document.ticker, Document.fiscal_year, Document.company_name)
                .order_by(Document.ticker, Document.fiscal_year.desc())
                .all()
            )
            return "\n".join([f"- {d[0]} (FY{d[1]} - {d[2]})" for d in available_docs])
        except Exception as e:
            logger.warning(f"Failed to fetch catalog summary: {e}")
            return ""
        finally:
            db.close()

    async def arespond(
        self,
        user_query: str,
        messages: Optional[List[Dict[str, str]]] = None,
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        """Generates a natural conversational response asynchronously."""
        catalog_summary = await asyncio.to_thread(self._get_catalog_summary)
        active_session = dict(session_state or {})

        system_prompt = render_prompt(
            "conversational",
            catalog_summary=catalog_summary,
            active_session=active_session,
        )

        chat_history = [SystemMessage(content=system_prompt)]
        if messages:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    if len(content) > 3000:
                        content = content[:3000] + "\n\n...[Prior research report excerpted for conversational context]..."
                    chat_history.append(AIMessage(content=content))

        if not chat_history or not isinstance(chat_history[-1], HumanMessage):
            chat_history.append(HumanMessage(content=user_query))

        llm = self._get_llm()
        llm_config = {"callbacks": callbacks} if callbacks else {}
        ai_response = await asyncio.to_thread(llm.invoke, chat_history, config=llm_config)
        return ai_response.content if hasattr(ai_response, "content") else str(ai_response)

    def respond(
        self,
        user_query: str,
        messages: Optional[List[Dict[str, str]]] = None,
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        """Synchronous wrapper for generating conversational responses."""
        return asyncio.run(
            self.arespond(
                user_query=user_query,
                messages=messages,
                session_state=session_state,
                callbacks=callbacks,
            )
        )

    def run(
        self,
        messages: List[Dict[str, str]],
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> AgentOutput:
        """BaseAgent run implementation."""
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        if not last_user_msg and messages:
            last_user_msg = messages[-1].get("content", "")

        answer = self.respond(
            user_query=last_user_msg,
            messages=messages,
            session_state=session_state,
            callbacks=callbacks,
        )
        return AgentOutput(
            content=answer,
            sources=[],
            updated_session_state=session_state,
        )
