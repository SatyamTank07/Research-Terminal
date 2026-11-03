"""Lead Supervisor & Intent Router Agent.

Responsible for query triage, filing catalog validation against the PostgreSQL
`documents` table, session context management, and establishing the execution
DAG (Fast-Path DCF vs. Full 10-K Equity Research Report vs. Single-Agent ad-hoc query).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.base import AgentOutput, BaseAgent
from app.agents.pipeline.contract import Conversation, RunContext, StatusLine
from app.agents.registry import AgentRegistry
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.supervisor.context import FilingCatalogError, active_session, load_filing, load_filing_catalog
from app.agents.specialized.supervisor.modules.company_resolution import CLARIFY_ENTITY_MESSAGE, resolve_company
from app.agents.specialized.supervisor.modules.filing_resolution import (
    choose_filing_year,
    explain_missing_filing,
    resolve_requested_year,
)
from app.agents.specialized.supervisor.modules.query_intent import SupervisorExtraction, verify_query_intent
from app.agents.specialized.supervisor.modules.valuation_parameters import (
    SESSION_KEY,
    parse_valuation_overrides,
    resolve_session_overrides,
    resolve_terminal_growth,
    resolve_valuation_parameters,
)
from app.agents.specialized.supervisor.state_supervisor import RoutingPlan, TurnResolution

logger = logging.getLogger("finance_agent.agents.supervisor")


@AgentRegistry.register("supervisor")
class SupervisorAgent(BaseAgent):
    """Lead Supervisor coordinating filing resolution, intent routing, and DAG planning."""

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
        """Utilizes supervisor.j2 prompt template for LLM routing and entity extraction with conversational history."""
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
        """Generates a complete RoutingPlan using database session_state for clean multi-turn context."""
        active_state = active_session(session_state)
        overrides = parse_valuation_overrides(user_query)

        # 1. Intent and entity resolution via LLM with full message history and active session context
        extraction: Optional[SupervisorExtraction] = None
        try:
            extraction = self._extract_with_llm(
                user_query,
                messages=messages,
                active_session=active_state,
                callbacks=callbacks,
            )
        except Exception as e:
            logger.warning(f"LLM supervisor extraction failed ({e})")

        intent = verify_query_intent(extraction)
        query_type, provenance = intent.query_type, intent.provenance

        # 2. Conversational queries are answered directly (single LLM hop)
        if query_type == "conversational":
            return self._conversational_plan(
                active_state, overrides, intent.reply, provenance,
                ticker=ticker or (extraction.extracted_ticker if extraction else None),
            )

        # 3. Ticker: explicit argument, then the LLM extraction, then the active session (deterministic fallback)
        notes: List[str] = list(overrides.notes)
        session_ticker = (active_state.get("active_ticker") or "").upper() or None
        company = resolve_company(ticker, extraction.extracted_ticker if extraction else None, session_ticker)
        if company is None:
            return self._conversational_plan(active_state, overrides, CLARIFY_ENTITY_MESSAGE, "deterministic_rule")
        resolved_ticker = company.ticker

        # 4. Filing: requested year (or the session's on a follow-up), checked against the ingested 10-Ks
        requested = resolve_requested_year(
            fiscal_year, extraction.extracted_year if extraction else None, user_query, resolved_ticker, active_state,
        )
        if requested.note:
            notes.append(requested.note)
        try:
            available_years = load_filing_catalog(resolved_ticker)
        except FilingCatalogError as err:
            return self._conversational_plan(
                active_state, overrides, explain_missing_filing(str(err)), "deterministic_rule",
            )
        choice = choose_filing_year(resolved_ticker, requested.year, user_query, available_years)
        filing = load_filing(resolved_ticker, choice.fiscal_year)
        doc_ticker, company_name, catalog_year, doc_id = (
            filing.ticker, filing.company_name, filing.fiscal_year, filing.document_id,
        )
        year_req, year_sub, avail_years = choice.year_requested, choice.year_substituted, available_years

        # 5. Valuation overrides: this turn's values, plus remembered ones for the same company
        parameters = resolve_valuation_parameters(overrides, active_state.get(SESSION_KEY), doc_ticker)
        valuation_overrides = parameters.overrides
        if parameters.carry_note:
            notes.append(parameters.carry_note)

        updated_session: Dict[str, Any] = {
            "active_ticker": doc_ticker,
            "active_company": company_name,
            "active_fiscal_year": catalog_year,
            "available_fiscal_years": avail_years,
            "last_query_type": query_type,
        }
        if valuation_overrides:
            updated_session[SESSION_KEY] = valuation_overrides

        logger.info(
            f"Supervisor routed query '{user_query[:50]}' -> "
            f"Ticker: {doc_ticker}, FY{catalog_year} (year_substituted={year_sub}, available_years={avail_years}), "
            f"Route: {query_type} ({provenance}), overrides={valuation_overrides or None}"
        )

        return RoutingPlan(
            ticker=doc_ticker,
            company_name=company_name,
            fiscal_year=catalog_year,
            available_fiscal_years=avail_years,
            year_requested=year_req,
            year_substituted=year_sub,
            document_id=doc_id,
            query_type=query_type,
            routing_provenance=provenance,
            conversational_response=None,
            valuation_overrides=valuation_overrides,
            parameter_notes=notes,
            updated_session_state=updated_session,
        )

    @staticmethod
    def _conversational_plan(
        active_state: Dict[str, Any],
        overrides,
        response: str,
        provenance: str,
        ticker: Optional[str] = None,
    ) -> RoutingPlan:
        """Direct reply that keeps the research context; parameters given now are remembered for the next run.

        Values stay bound to the active company, or unbound (attach to the next company named) when none is active.
        """
        session = dict(active_state)
        remembered = resolve_session_overrides(overrides, session.get(SESSION_KEY), session.get("active_ticker"))
        if remembered:
            session[SESSION_KEY] = remembered
        else:
            session.pop(SESSION_KEY, None)
        active_ticker = ticker or session.get("active_ticker") or ""
        return RoutingPlan(
            ticker=active_ticker,
            company_name=session.get("active_company") or active_ticker or None,
            fiscal_year=session.get("active_fiscal_year") or 0,
            available_fiscal_years=session.get("available_fiscal_years") or [],
            document_id=None,
            query_type="conversational",
            routing_provenance=provenance,
            conversational_response=response,
            parameter_notes=list(overrides.notes),
            updated_session_state=session,
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
            if plan.query_type == "conversational" and plan.conversational_response:
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
    logger.info(
        f"[supervisor] Resolved {plan.ticker} FY{plan.fiscal_year} "
        f"route={plan.query_type} (year_substituted={plan.year_substituted})"
    )
    run_context = None
    if plan.query_type != "conversational":
        run_context = RunContext(
            ticker=plan.ticker,
            company_name=plan.company_name or plan.ticker,
            fiscal_year=plan.fiscal_year,
            available_fiscal_years=tuple(plan.available_fiscal_years),
            document_id=plan.document_id,
            user_query=conversation.user_query,
            route=plan.query_type,
            year_substituted=plan.year_substituted,
            valuation_overrides=dict(plan.valuation_overrides),
            terminal_growth_rate=resolve_terminal_growth(plan.valuation_overrides),
            valuation_date=conversation.run_date,
            callbacks=conversation.callbacks,
        )
    return TurnResolution(
        route=plan.query_type,
        plan=plan,
        run_context=run_context,
        conversational_response=plan.conversational_response,
        session_update=plan.updated_session_state,
        parameter_notes=tuple(plan.parameter_notes),
    )


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
