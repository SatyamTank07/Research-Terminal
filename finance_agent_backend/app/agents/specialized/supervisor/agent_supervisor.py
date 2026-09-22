"""Lead Supervisor & Intent Router Agent.

Responsible for deterministic query triage, filing catalog validation against
the PostgreSQL `documents` table, and establishing the execution DAG
(Fast-Path DCF vs. Full 10-K Equity Research Report vs. Single-Agent ad-hoc query).
Includes explicit provenance tracking (substitution flag, routing provenance)
and an LLM-assisted fallback using supervisor.j2 for ambiguous natural language queries.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.base import AgentOutput, BaseAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.registry import AgentRegistry
from app.agents.state import QueryType, RoutingPlan
from app.database import SessionLocal
from app.models import Document

logger = logging.getLogger("finance_agent.agents.supervisor")

# Comprehensive financial, corporate, executive, and English stopwords to prevent ticker false-positives
STOPWORD_TICKERS = {
    # Pronouns, Conjunctions & Prepositions
    "A", "AN", "THE", "AND", "OR", "BUT", "IF", "SO", "FOR", "AT", "BY", "FROM",
    "IN", "INTO", "OF", "OFF", "ON", "ONTO", "OUT", "OVER", "TO", "UP", "WITH",
    "AS", "BE", "IS", "ARE", "WAS", "WERE", "DO", "DOES", "DID", "HAVE", "HAS",
    "HAD", "CAN", "COULD", "WILL", "WOULD", "SHALL", "SHOULD", "MAY", "MIGHT",
    "MUST", "I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "HIM", "HER",
    "US", "THEM", "MY", "YOUR", "HIS", "ITS", "OUR", "THEIR", "WHAT", "WHICH",
    "WHO", "WHOM", "WHOSE", "WHEN", "WHERE", "WHY", "HOW", "ALL", "ANY", "BOTH",
    "EACH", "FEW", "MORE", "MOST", "OTHER", "SOME", "SUCH", "NO", "NOR", "NOT",
    "ONLY", "OWN", "SAME", "THAN", "TOO", "VERY", "JUST",
    # Financial, SEC, Executive & Filing Acronyms
    "10K", "10-K", "10Q", "10-Q", "8K", "8-K", "FY", "Q1", "Q2", "Q3", "Q4",
    "SEC", "EDGAR", "GAAP", "IFRS", "EPS", "PE", "PB", "PS", "EV",
    "EBITDA", "EBIT", "NOPAT", "FCF", "UFCF", "CAGR", "YOY", "QOQ", "TTM",
    "LTM", "NTM", "WACC", "ROIC", "ROE", "ROA", "ROC", "IRR", "NPV", "BPS",
    "SGA", "SG&A", "RD", "R&D", "CAPEX", "OPEX", "COGS", "NWC", "MD&A",
    "CEO", "CFO", "COO", "CTO", "CIO", "CMO", "VP", "SVP", "EVP", "USA", "USD",
    # Generic Financial & Agent Terms
    "SHOW", "FIND", "RUN", "GET", "GIVE", "TELL", "CHECK", "REPORT", "STOCK",
    "PRICE", "VALUE", "MODEL", "ANALYSIS", "AGENT", "AGENTS", "VIEW", "CALC",
    "CALCULATE", "SUMMARIZE", "EXPLAIN", "AUDIT", "FORECAST", "VALUATION",
    "RESEARCH", "COMPANY", "CORP", "INC", "LTD", "FILING", "STATEMENT", "TABLE",
}

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
}


class LLMRoutingDecision(BaseModel):
    """Pydantic schema for supervisor.j2 LLM fallback classification."""

    query_type: QueryType = Field(..., description="Selected routing path")
    extracted_ticker: Optional[str] = Field(None, description="Extracted stock ticker if identifiable")
    extracted_year: Optional[int] = Field(None, description="Extracted fiscal year if identifiable")
    routing_reasoning: str = Field(..., description="Brief explanation of the routing classification")


@AgentRegistry.register("supervisor")
class SupervisorAgent(BaseAgent):
    """Lead Supervisor coordinating filing resolution, deterministic routing, and DAG planning."""

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

    def resolve_filing_catalog(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        user_query: Optional[str] = None,
    ) -> Tuple[str, str, int, str, Optional[int], bool]:
        """Resolves target company, filing year, and document ID from database catalog.

        Explicitly tracks whether requested year was substituted with an audited alternative.

        Returns:
            Tuple of (ticker, company_name, fiscal_year, document_id, year_requested, year_substituted)
        """
        db = SessionLocal()
        try:
            resolved_ticker = (ticker or "").strip().upper()
            year_requested = fiscal_year

            # If ticker not explicitly provided, attempt extraction from user query
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
                    # Requested year unavailable; fall back to latest available year with explicit flag
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

        # 2. Company name substring match against documents
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

        # 3. Uppercase ticker pattern fallback filtering out extensive financial stopwords
        candidates = re.findall(r"\b[A-Z]{1,5}\b", query)
        for cand in candidates:
            if cand not in STOPWORD_TICKERS:
                return cand

        return ""

    def classify_intent(self, user_query: str) -> QueryType:
        """Public API returning purely the QueryType."""
        query_type, _ = self.classify_intent_with_provenance(user_query)
        return query_type

    def classify_intent_with_provenance(self, user_query: str) -> Tuple[QueryType, str]:
        """Classifies user intent into execution routes with refined keyword prioritization.

        Returns:
            Tuple of (QueryType, routing_provenance: "deterministic_rule" | "llm_inferred")
        """
        q = user_query.lower()

        # 1. Check for explicit full report request
        full_report_signals = [
            "full report", "complete report", "comprehensive", "deep dive",
            "full 10-k", "full 10k", "equity research report", "full research",
            "institutional report", "all agents", "overall analysis"
        ]
        if any(sig in q for sig in full_report_signals):
            return "full_10k_report", "deterministic_rule"

        # 2. Check for Moat / Business Strategy questions FIRST (prevents broad "valuation" words from hijacking)
        moat_signals = [
            "moat", "economic moat", "business model", "competitive advantage",
            "segments", "product segments", "pricing power", "customer concentration",
            "revenue model", "how does it make money", "ecosystem advantage"
        ]
        if any(sig in q for sig in moat_signals):
            return "business_moat_only", "deterministic_rule"

        # 3. Check for Risk Factors questions
        risk_signals = [
            "risk", "risks", "risk factor", "risk factors", "threat", "threats",
            "litigation", "lawsuit", "antitrust", "regulatory threat",
            "existential threat", "headwinds", "vulnerabilities"
        ]
        if any(sig in q for sig in risk_signals):
            return "risk_factors_only", "deterministic_rule"

        # 4. Check for Financial Statement / Auditor questions
        audit_signals = [
            "balance sheet", "income statement", "cash flow statement",
            "statement of operations", "statement of cash flows", "gross margin",
            "operating margin", "net margin", "audit", "financial ratios",
            "forensic", "red flags", "net debt", "shares outstanding"
        ]
        if any(sig in q for sig in audit_signals):
            return "financial_audit_only", "deterministic_rule"

        # 5. Check for focused DCF Valuation
        dcf_signals = [
            "dcf", "discounted cash flow", "intrinsic value", "fair value",
            "target price", "wacc", "cost of capital", "what is the fair value",
            "what is the intrinsic value", "sensitivity matrix", "valuation multiple"
        ]
        if any(sig in q for sig in dcf_signals):
            return "dcf_valuation_only", "deterministic_rule"

        # Standalone "valuation" only routes to DCF if explicitly numerical
        if "valuation" in q and not any(sig in q for sig in moat_signals):
            return "dcf_valuation_only", "deterministic_rule"

        # 6. Fallback: For conversational/ambiguous queries, invoke supervisor.j2 LLM router
        try:
            llm_decision = self._classify_with_llm(user_query)
            if llm_decision and llm_decision.query_type in AGENT_EXECUTION_PLANS:
                return llm_decision.query_type, "llm_inferred"
        except Exception as e:
            logger.warning(f"LLM supervisor intent routing failed ({e}); defaulting to full_10k_report.")

        # Default for broad inquiries (e.g. "Analyze Apple", "TSLA research")
        return "full_10k_report", "deterministic_rule"

    def _classify_with_llm(self, user_query: str) -> Optional[LLMRoutingDecision]:
        """Utilizes supervisor.j2 prompt template for LLM routing and entity extraction on ambiguous queries."""
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

        system_prompt = render_prompt("supervisor", catalog_summary=catalog_summary)
        llm = self._get_llm()
        structured_llm = llm.with_structured_output(LLMRoutingDecision)

        try:
            decision: LLMRoutingDecision = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze and route this research inquiry: '{user_query}'"),
            ])
            return decision
        except Exception as e:
            logger.warning(f"LLM supervisor routing failed: {e}")
            return None

    def route(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
    ) -> RoutingPlan:
        """Generates a complete RoutingPlan with integrated deterministic and LLM entity/intent resolution."""
        # 1. Attempt deterministic catalog resolution first
        try:
            resolved_ticker, company_name, resolved_year, doc_id, year_req, year_sub = (
                self.resolve_filing_catalog(
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    user_query=user_query,
                )
            )
            query_type, provenance = self.classify_intent_with_provenance(user_query)
        except ValueError as deterministic_err:
            # 2. Deterministic ticker extraction failed -> invoke supervisor.j2 LLM fallback!
            logger.info(
                f"Deterministic catalog resolution failed ({deterministic_err}). "
                f"Attempting supervisor.j2 LLM entity and intent resolution."
            )
            llm_decision = self._classify_with_llm(user_query)
            if llm_decision and llm_decision.extracted_ticker:
                resolved_ticker, company_name, resolved_year, doc_id, year_req, year_sub = (
                    self.resolve_filing_catalog(
                        ticker=llm_decision.extracted_ticker,
                        fiscal_year=llm_decision.extracted_year or fiscal_year,
                        user_query=user_query,
                    )
                )
                query_type = llm_decision.query_type
                provenance = "llm_inferred"
            else:
                # Both deterministic and LLM extraction failed; re-raise original descriptive error
                raise deterministic_err

        active_agents = AGENT_EXECUTION_PLANS.get(query_type, AGENT_EXECUTION_PLANS["full_10k_report"])

        logger.info(
            f"Supervisor routed query '{user_query[:50]}' -> "
            f"Ticker: {resolved_ticker}, FY{resolved_year} (Substituted: {year_sub}), "
            f"Route: {query_type} ({provenance}), Agents: {active_agents}"
        )

        return RoutingPlan(
            ticker=resolved_ticker,
            company_name=company_name,
            fiscal_year=resolved_year,
            year_requested=year_req,
            year_substituted=year_sub,
            document_id=doc_id,
            query_type=query_type,
            active_agents=active_agents,
            routing_provenance=provenance,
        )

    def run(self, messages: List[Dict[str, str]]) -> AgentOutput:
        """Executes the supervisor on conversational messages conforming to BaseAgent."""
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        if not last_user_msg and messages:
            last_user_msg = messages[-1].get("content", "")

        try:
            plan = self.route(user_query=last_user_msg)
            return AgentOutput(
                content=json.dumps(plan.model_dump(), indent=2),
                sources=[{
                    "document_id": plan.document_id,
                    "ticker": plan.ticker,
                    "fiscal_year": plan.fiscal_year,
                    "year_substituted": plan.year_substituted,
                    "routing_provenance": plan.routing_provenance,
                }],
            )
        except ValueError as e:
            return AgentOutput(
                content=f"Supervisor Routing Error: {str(e)}",
                sources=[],
            )
