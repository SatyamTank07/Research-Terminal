"""Lead Supervisor & Intent Router Agent.

Responsible for query triage, filing catalog validation against the PostgreSQL
`documents` table, session context management, and establishing the execution
DAG (Fast-Path DCF vs. Full 10-K Equity Research Report vs. Single-Agent ad-hoc query).
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.base import AgentOutput, BaseAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.registry import AgentRegistry
from app.agents.specialized.supervisor.state_supervisor import QueryType, RoutingPlan
from app.database import SessionLocal

from app.models import Document

logger = logging.getLogger("finance_agent.agents.supervisor")

# Deprecated: Kept as empty set for backwards compatibility with external callers
STOPWORD_TICKERS: set = set()

# Fast-path agent mapping
AGENT_EXECUTION_PLANS: Dict[QueryType, List[str]] = {
    "full_10k_report": [
        "business_strategist",
        "financial_auditor",
        "risk_analyst",
        "forecasting_analyst",
        "valuation_specialist",
        "lead_synthesizer",
    ],
    "dcf_valuation_only": [
        "financial_auditor",
        "forecasting_analyst",
        "valuation_specialist",
    ],
    "financial_audit_only": [
        "financial_auditor",
    ],
    "business_moat_only": [
        "business_strategist",
    ],
    "risk_factors_only": [
        "risk_analyst",
    ],
    "conversational": [],
}


class SupervisorExtraction(BaseModel):
    """Pydantic schema for supervisor intent and entity extraction."""

    query_type: QueryType = Field(
        default="conversational",
        description="Selected routing path (full_10k_report, dcf_valuation_only, financial_audit_only, business_moat_only, risk_factors_only, conversational)",
    )
    extracted_ticker: Optional[str] = Field(
        None, description="Extracted stock ticker if identifiable (e.g. AAPL, TSLA, NVDA)"
    )
    extracted_year: Optional[int] = Field(
        None, description="Extracted 4-digit fiscal year if explicitly requested by user (e.g. 2023)"
    )
    conversational_response: Optional[str] = Field(
        None,
        description=(
            "If query_type == 'conversational', provide the full, professional response "
            "directly to the user as a Senior Equity Research Director. If query requires 10-K execution, leave null."
        ),
    )
    routing_reasoning: str = Field(
        default="", description="Brief explanation of the routing classification"
    )


# Alias for backward compatibility
LLMRoutingDecision = SupervisorExtraction


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
        active_session: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> SupervisorExtraction:
        """Utilizes supervisor.j2 prompt template for LLM routing and entity extraction with active session context."""
        db = SessionLocal()
        try:
            available_docs = (
                db.query(Document.ticker, Document.fiscal_year, Document.company_name)
                .order_by(Document.ticker, Document.fiscal_year.desc())
                .all()
            )
            catalog_summary = "\n".join([f"- {d[0]} (FY{d[1]}): {d[2]}" for d in available_docs])
        finally:
            db.close()

        system_prompt = render_prompt("supervisor", catalog_summary=catalog_summary, active_session=active_session)
        llm = self._get_llm()
        structured_llm = llm.with_structured_output(SupervisorExtraction)

        prompt_input = f"Analyze and route this research inquiry: '{user_query}'"
        llm_config = {"callbacks": callbacks} if callbacks else {}

        decision: SupervisorExtraction = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_input),
        ], config=llm_config)
        return decision

    def resolve_filing_catalog(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        user_query: Optional[str] = None,
    ) -> Tuple[str, str, int, str, Optional[int], bool]:
        """Resolves target company, filing year, and document ID from database catalog.

        Returns:
            Tuple of (ticker, company_name, fiscal_year, document_id, year_requested, year_substituted)
        """
        db = SessionLocal()
        try:
            resolved_ticker = (ticker or "").strip().upper()
            year_requested = fiscal_year

            # If ticker not explicitly provided, attempt extraction from query
            if not resolved_ticker and user_query:
                resolved_ticker = self._extract_ticker_from_query(user_query, db)

            # If fiscal year not explicitly provided, attempt extraction from query
            if not year_requested and user_query:
                year_match = re.search(r"\b(20[12]\d)\b", user_query)
                if year_match:
                    year_requested = int(year_match.group(1))

            if not resolved_ticker:
                available_docs = (
                    db.query(Document.ticker, Document.fiscal_year, Document.company_name)
                    .order_by(Document.ticker, Document.fiscal_year.desc())
                    .all()
                )
                available_str = ", ".join([f"{d[0]} (FY{d[1]} - {d[2]})" for d in available_docs])
                raise ValueError(
                    f"Could not identify a target stock ticker from the request. "
                    f"Available ingested filings in database: [{available_str}]"
                )

            # Look up document by ticker
            query = db.query(Document).filter(Document.ticker == resolved_ticker)
            year_substituted = False

            if year_requested:
                doc = query.filter(Document.fiscal_year == year_requested).first()
                if not doc:
                    # Requested year unavailable; select latest available year with substitution flag
                    doc = (
                        db.query(Document)
                        .filter(Document.ticker == resolved_ticker)
                        .order_by(Document.fiscal_year.desc())
                        .first()
                    )
                    if doc:
                        year_substituted = True
                        logger.warning(
                            f"Requested filing FY{year_requested} for '{resolved_ticker}' not found in database. "
                            f"Substituted with latest audited filing FY{doc.fiscal_year} (year_substituted=True)."
                        )
            else:
                doc = query.order_by(Document.fiscal_year.desc()).first()

            if not doc:
                available_tickers = [t[0] for t in db.query(Document.ticker).distinct().all()]
                raise ValueError(
                    f"No ingested 10-K filings found for ticker '{resolved_ticker}'. "
                    f"Available tickers in database: {available_tickers}"
                )

            return doc.ticker, doc.company_name, doc.fiscal_year, doc.id, year_requested, year_substituted

        finally:
            db.close()

    def _extract_ticker_from_query(self, query: str, db: Any) -> str:
        """Extracts ticker symbol or matches company name against documents in database."""
        # 1. Exact match against known database tickers
        available_tickers = {t[0].upper() for t in db.query(Document.ticker).all()}
        tokens = re.findall(r"\b[A-Za-z0-9]+\b", query)
        for token in tokens:
            t_upper = token.upper()
            if t_upper in available_tickers:
                return t_upper

        # 2. Company name match against documents
        all_docs = db.query(Document.ticker, Document.company_name).all()
        q_lower = query.lower()
        for t, cname in all_docs:
            clean_cname = re.sub(
                r"\b(inc|corp|corporation|ltd|limited|co|company|holdings|plc|nv)\b|[,\.]",
                "",
                cname.lower(),
                flags=re.IGNORECASE,
            ).strip()
            first_word = clean_cname.split()[0] if clean_cname else ""
            if first_word and len(first_word) >= 3 and first_word in q_lower:
                return t.upper()

        return ""

    def classify_intent(self, user_query: str) -> QueryType:
        """Public API returning purely the QueryType."""
        query_type, _ = self.classify_intent_with_provenance(user_query)
        return query_type

    def classify_intent_with_provenance(
        self,
        user_query: str,
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Tuple[QueryType, str]:
        """Classifies user intent using LLM inference driven by prompt_supervisor.j2."""
        try:
            extraction = self._extract_with_llm(
                user_query, active_session=session_state, callbacks=callbacks
            )
            if extraction and extraction.query_type in AGENT_EXECUTION_PLANS:
                return extraction.query_type, "llm_inferred"
        except Exception as e:
            logger.warning(f"LLM supervisor intent routing failed ({e}); defaulting to conversational.")

        return "conversational", "deterministic_rule"

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
        active_state = dict(session_state or {})

        # 1. Clean up any obsolete pending_action from legacy session state
        if "pending_action" in active_state:
            active_state["pending_action"] = None

        # 2. Intent and entity resolution via LLM with active session context
        extraction: Optional[SupervisorExtraction] = None
        try:
            extraction = self._extract_with_llm(user_query, active_session=active_state, callbacks=callbacks)
        except Exception as e:
            logger.warning(f"LLM supervisor extraction failed ({e})")

        if extraction and extraction.query_type in AGENT_EXECUTION_PLANS:
            query_type = extraction.query_type
            provenance = "llm_inferred"
        else:
            query_type = "conversational"
            provenance = "deterministic_rule"

        # Handle conversational queries directly (Single LLM Hop)
        if query_type == "conversational":
            active_ticker = (
                ticker
                or (extraction.extracted_ticker if extraction else None)
                or active_state.get("active_ticker")
            )
            active_company = active_state.get("active_company") or active_ticker
            active_year = active_state.get("active_fiscal_year") or 0

            conv_response = (
                extraction.conversational_response
                if (extraction and extraction.conversational_response)
                else (
                    "I am your Senior Equity Research Director. I can assist you with comprehensive SEC 10-K research reports, "
                    "three-statement financial audits, 5-year UFCF projections, and Gordon Growth DCF valuations. "
                    "Which company or ticker would you like to evaluate?"
                )
            )

            return RoutingPlan(
                ticker=active_ticker or "",
                company_name=active_company or (active_ticker or "Senior Equity Research Director"),
                fiscal_year=active_year or 0,
                year_requested=None,
                year_substituted=False,
                document_id=None,
                query_type="conversational",
                active_agents=[],
                routing_provenance=provenance,
                conversational_response=conv_response,
                needs_confirmation=False,
                confirmation_message=None,
                suggested_fiscal_year=None,
                updated_session_state=active_state,
            )

        resolved_ticker = (
            ticker
            or (extraction.extracted_ticker if extraction else None)
            or active_state.get("active_ticker")
        )
        resolved_year_req = (
            fiscal_year if fiscal_year is not None else (extraction.extracted_year if extraction else None)
        )

        try:
            doc_ticker, company_name, catalog_year, doc_id, year_req, year_sub = (
                self.resolve_filing_catalog(
                    ticker=resolved_ticker,
                    fiscal_year=resolved_year_req,
                    user_query=user_query,
                )
            )
        except ValueError as err:
            if extraction and extraction.extracted_ticker and extraction.extracted_ticker != resolved_ticker:
                doc_ticker, company_name, catalog_year, doc_id, year_req, year_sub = (
                    self.resolve_filing_catalog(
                        ticker=extraction.extracted_ticker,
                        fiscal_year=extraction.extracted_year or fiscal_year,
                        user_query=user_query,
                    )
                )
            else:
                # If ticker could not be resolved from query at all, gracefully route to conversational
                if not resolved_ticker:
                    return RoutingPlan(
                        ticker="",
                        company_name="Senior Equity Research Director",
                        fiscal_year=0,
                        year_requested=None,
                        year_substituted=False,
                        document_id=None,
                        query_type="conversational",
                        active_agents=[],
                        routing_provenance="deterministic_rule",
                        conversational_response=(
                            "I am your Senior Equity Research Director. I could not identify an ingested company filing "
                            "matching your request. Please specify a ticker symbol (e.g. AAPL, TSLA, NVDA) or upload a 10-K filing."
                        ),
                        needs_confirmation=False,
                        confirmation_message=None,
                        suggested_fiscal_year=None,
                        updated_session_state=active_state,
                    )
                raise err

        # 3. Schedule active agents with automatic year substitution (Zero-Halting)
        active_agents = AGENT_EXECUTION_PLANS.get(query_type, AGENT_EXECUTION_PLANS["full_10k_report"])
        updated_session = {
            "active_ticker": doc_ticker,
            "active_company": company_name,
            "active_fiscal_year": catalog_year,
            "last_query_type": query_type,
            "pending_action": None,
        }

        logger.info(
            f"Supervisor routed query '{user_query[:50]}' -> "
            f"Ticker: {doc_ticker}, FY{catalog_year} (year_substituted={year_sub}), "
            f"Route: {query_type} ({provenance})"
        )

        return RoutingPlan(
            ticker=doc_ticker,
            company_name=company_name,
            fiscal_year=catalog_year,
            year_requested=year_req,
            year_substituted=year_sub,
            document_id=doc_id,
            query_type=query_type,
            active_agents=active_agents,
            routing_provenance=provenance,
            conversational_response=None,
            needs_confirmation=False,
            confirmation_message=None,
            suggested_fiscal_year=None,
            updated_session_state=updated_session,
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
