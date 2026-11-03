"""Supervisor: turns a chat message into a route and the facts every agent of the run shares.

`SupervisorAgent.route` is the checklist: one judge call (`_extract_with_llm`), then the modules in
`MODULES` order, with the catalog read from `context.py` and the plan built by `assemble.py`.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.base import AgentOutput, BaseAgent
from app.agents.pipeline.contract import Conversation, StatusLine
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.supervisor.assemble import (
    assemble_conversational_plan,
    assemble_research_plan,
    assemble_turn,
)
from app.agents.specialized.supervisor.context import (
    FilingCatalogError,
    active_session,
    load_filing,
    load_filing_catalog,
)
from app.agents.specialized.supervisor.modules.company_resolution import CLARIFY_ENTITY_MESSAGE, resolve_company
from app.agents.specialized.supervisor.modules.filing_resolution import (
    choose_filing_year,
    explain_missing_filing,
    resolve_requested_year,
)
from app.agents.specialized.supervisor.modules.query_intent import (
    CONVERSATIONAL,
    SupervisorExtraction,
    verify_query_intent,
)
from app.agents.specialized.supervisor.modules.session_continuity import SessionFiling, remember_turn
from app.agents.specialized.supervisor.modules.valuation_parameters import (
    SESSION_KEY,
    ValuationOverrides,
    parse_valuation_overrides,
    resolve_valuation_parameters,
)
from app.agents.specialized.supervisor.state_supervisor import RoutingPlan, TurnResolution

logger = logging.getLogger("finance_agent.agents.supervisor")


class SupervisorAgent(BaseAgent):
    """Lead Supervisor: intent, company, filing and valuation-parameter resolution for one chat turn."""

    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        self.model_name = model_name
        self._cached_llm = None

    def _get_llm(self):
        """Lazy loader with cache-first check to prevent redundant disk I/O."""
        if self._cached_llm is not None:
            return self._cached_llm

        load_dotenv(override=True)
        model_clean = self.model_name.replace("openai:", "")
        llm = ChatOpenAI(
            model=model_clean,
            temperature=0,
            max_retries=3,
        )
        self._cached_llm = llm
        return llm

    def _extract_with_llm(
        self,
        user_query: str,
        messages: Optional[List[Dict[str, str]]] = None,
        active_session: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> SupervisorExtraction:
        """Utilizes the prompt_supervisor.j2 template for LLM routing and entity extraction with conversational history."""
        system_prompt = render_prompt("supervisor", "prompt_supervisor.j2", active_session=active_session)
        llm = self._get_llm()
        structured_llm = llm.with_structured_output(SupervisorExtraction)

        messages_payload = [SystemMessage(content=system_prompt)]

        # Inject past conversation history for full multi-turn context
        if messages and len(messages) > 1:
            for m in messages[:-1]:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "user":
                    messages_payload.append(HumanMessage(content=content))
                elif role == "assistant":
                    # Truncate prior long reports to save tokens while preserving conversational context
                    if len(content) > 1500:
                        content = content[:1500] + "\n...[Prior research report excerpted]..."
                    messages_payload.append(AIMessage(content=content))

        # Append the latest user inquiry
        messages_payload.append(HumanMessage(content=f"Analyze and route this research inquiry: '{user_query}'"))
        llm_config = {"callbacks": callbacks} if callbacks else {}

        decision: SupervisorExtraction = structured_llm.invoke(messages_payload, config=llm_config)
        return decision

    def route(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        session_state: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> RoutingPlan:
        """The Supervisor's checklist: intent -> company -> filing -> valuation parameters -> session -> plan."""
        session = active_session(session_state)
        overrides = parse_valuation_overrides(user_query)

        extraction: Optional[SupervisorExtraction] = None
        try:
            extraction = self._extract_with_llm(
                user_query,
                messages=messages,
                active_session=session,
                callbacks=callbacks,
            )
        except Exception as e:
            logger.warning(f"LLM supervisor extraction failed ({e})")
        intent = verify_query_intent(extraction)
        if intent.query_type == CONVERSATIONAL:
            return _reply_in_chat(
                session, overrides, intent.reply, intent.provenance,
                ticker=ticker or (extraction.extracted_ticker if extraction else None),
            )

        session_ticker = (session.get("active_ticker") or "").upper() or None
        company = resolve_company(ticker, extraction.extracted_ticker if extraction else None, session_ticker)
        if company is None:
            return _reply_in_chat(session, overrides, CLARIFY_ENTITY_MESSAGE, "deterministic_rule")

        requested = resolve_requested_year(
            fiscal_year, extraction.extracted_year if extraction else None, user_query, company.ticker, session,
        )
        try:
            available_years = load_filing_catalog(company.ticker)
        except FilingCatalogError as err:
            return _reply_in_chat(session, overrides, explain_missing_filing(str(err)), "deterministic_rule")
        choice = choose_filing_year(company.ticker, requested.year, user_query, available_years)
        filing = load_filing(company.ticker, choice.fiscal_year)

        parameters = resolve_valuation_parameters(overrides, session.get(SESSION_KEY), filing.ticker)
        remembered = remember_turn(
            session,
            parameters.overrides,
            SessionFiling(ticker=filing.ticker, company_name=filing.company_name, fiscal_year=filing.fiscal_year,
                          available_fiscal_years=available_years),
            intent.query_type,
        )

        logger.info(
            f"Supervisor routed query '{user_query[:50]}' -> "
            f"Ticker: {filing.ticker}, FY{filing.fiscal_year} (year_substituted={choice.year_substituted}, "
            f"available_years={available_years}), "
            f"Route: {intent.query_type} ({intent.provenance}), overrides={parameters.overrides or None}"
        )
        return assemble_research_plan(
            intent=intent, filing=filing, choice=choice, available_years=available_years,
            parse_notes=overrides.notes, filing_note=requested.note, parameters=parameters, session=remembered,
        )

    def run(
        self,
        messages: List[Dict[str, str]],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Executes the supervisor on conversational messages conforming to BaseAgent."""
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        if not last_user_msg and messages:
            last_user_msg = messages[-1].get("content", "")

        try:
            plan = self.route(
                user_query=last_user_msg,
                messages=messages,
                session_state=session_state,
            )
            if plan.query_type == CONVERSATIONAL and plan.conversational_response:
                return AgentOutput(
                    content=plan.conversational_response,
                    sources=[],
                    updated_session_state=plan.updated_session_state,
                )

            return AgentOutput(
                content=json.dumps(plan.model_dump(), indent=2),
                sources=[{
                    "document_id": plan.document_id,
                    "ticker": plan.ticker,
                    "fiscal_year": plan.fiscal_year,
                    "available_fiscal_years": plan.available_fiscal_years,
                    "year_substituted": plan.year_substituted,
                    "routing_provenance": plan.routing_provenance,
                }],
                updated_session_state=plan.updated_session_state,
            )
        except ValueError as e:
            return AgentOutput(
                content=f"Supervisor Routing Error: {str(e)}",
                sources=[],
            )


def _reply_in_chat(
    session: Dict[str, Any], overrides: ValuationOverrides, reply: str, provenance: str, ticker: Optional[str] = None,
) -> RoutingPlan:
    """A direct reply keeps the research context; parameters given now are remembered for the next run.

    Values stay bound to the active company, or unbound (attach to the next company named) when none is active.
    """
    parameters = resolve_valuation_parameters(overrides, session.get(SESSION_KEY), session.get("active_ticker"))
    return assemble_conversational_plan(
        remember_turn(session, parameters.overrides), reply, provenance, overrides.notes, ticker=ticker,
    )


def resolve_turn(conversation: Conversation) -> TurnResolution:
    """Routes one chat turn and, for a research route, fixes the facts every agent will share."""
    plan = SupervisorAgent().route(
        user_query=conversation.user_query,
        ticker=conversation.ticker,
        fiscal_year=conversation.fiscal_year,
        session_state=conversation.session_state,
        messages=conversation.messages,
        callbacks=conversation.callbacks,
    )
    return assemble_turn(conversation, plan)


def supervisor_milestone(turn: Optional[TurnResolution]) -> StatusLine:
    """SSE status line for a resolved research turn; conversational turns show none."""
    if turn is None or turn.run_context is None:
        return None, None
    plan = turn.plan
    sub_note = " (substituted year)" if plan.year_substituted else ""
    return (
        f"Resolved filing: {plan.ticker} FY{plan.fiscal_year}{sub_note} | Route: {plan.query_type}",
        {
            "ticker": plan.ticker,
            "fiscal_year": plan.fiscal_year,
            "query_type": plan.query_type,
            "year_substituted": plan.year_substituted,
            "valuation_overrides": plan.valuation_overrides,
        },
    )
