"""LangGraph Multi-Agent Orchestrator & State Machine.

Coordinates the end-to-end institutional equity research workflow across:
1. SupervisorAgent (triage, filing resolution, entity extraction, execution planning)
2. BusinessStrategistAgent (Item 1 qualitative business model & economic moat)
3. FinancialAuditorAgent (Item 8 audited financial statements, ratios, red flags)
4. RiskAnalystAgent (Item 1A material risks & existential threats)
5. ForecastingAnalystAgent (Item 7 MD&A 5-year revenue & UFCF projections)
6. ValuationSpecialistAgent (CAPM WACC hurdle rate, 2-stage Gordon Growth DCF, 5x5 sensitivity)
7. LeadSynthesizerAgent (3-Pillar Investment Thesis, executive summary, institutional publication)

Implements:
- Non-blocking asynchronous execution using LangGraph StateGraph.
- 5 execution routes (full_10k_report, dcf_valuation_only, financial_audit_only, business_moat_only, risk_factors_only).
- Parallel fan-out execution for Phase 1 qualitative and statement auditing.
- Milestone status event streaming for real-time UI updates.
"""

import asyncio
import concurrent.futures
import json
import logging
import re
from datetime import date
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.base import AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.business_strategist import BusinessStrategistAgent
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
from app.agents.specialized.lead_synthesizer import LeadSynthesizerAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.risk_analyst import RiskAnalystAgent
from app.agents.specialized.supervisor import SupervisorAgent
from app.agents.specialized.valuation_specialist import ValuationSpecialistAgent
from app.agents.state import (
    BusinessMoatOutput,
    DCFValuationOutput,
    EquityResearchState,
    Final10KResearchReport,
    FinancialAuditOutput,
    ForecastOutput,
    RiskAuditOutput,
    RoutingPlan,
)
from app.agents.tools.forecast_drivers import DEFAULT_TERMINAL_GROWTH
from app.agents.tools.market_data_tools import fetch_market_context
from app.agents.tools.valuation_model import MarketInputs
from app.config import settings
from app.database import SessionLocal
from app.models import ANNUAL_REPORT_FILING_TYPE, Document

logger = logging.getLogger("finance_agent.agents.orchestrator")


# ==============================================================================
# 1. Individual Node Execution Functions (Non-blocking Async Wrappers)
# ==============================================================================
async def supervisor_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 0: Triages user inquiry, resolves filing catalog, and emits RoutingPlan."""
    user_query = state.get("user_query", "")
    ticker = state.get("ticker")
    fiscal_year = state.get("fiscal_year")
    messages = state.get("messages")
    session_state = state.get("session_state")

    callbacks = state.get("callbacks")
    supervisor = SupervisorAgent()
    routing_plan: RoutingPlan = await asyncio.to_thread(
        supervisor.route,
        user_query=user_query,
        ticker=ticker,
        fiscal_year=fiscal_year,
        session_state=session_state,
        messages=messages,
        callbacks=callbacks,
    )

    logger.info(
        f"[supervisor_node] Resolved {routing_plan.ticker} FY{routing_plan.fiscal_year} "
        f"route={routing_plan.query_type} (year_substituted={routing_plan.year_substituted})"
    )

    return {
        "ticker": routing_plan.ticker,
        "company_name": routing_plan.company_name or routing_plan.ticker,
        "fiscal_year": routing_plan.fiscal_year,
        "available_fiscal_years": routing_plan.available_fiscal_years,
        "document_id": routing_plan.document_id,
        "query_type": routing_plan.query_type,
        "routing_plan": routing_plan,
        "conversational_response": routing_plan.conversational_response,
        "error_message": None,
        "updated_session_state": routing_plan.updated_session_state,
    }


async def business_strategist_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 1: Evaluates business model, product segments, and economic moat (Item 1)."""
    ticker = state["ticker"]
    fiscal_year = state["fiscal_year"]
    callbacks = state.get("callbacks")

    logger.info(f"[business_strategist_node] Analyzing {ticker} FY{fiscal_year}")
    strategist = BusinessStrategistAgent()
    moat_output: BusinessMoatOutput = await asyncio.to_thread(
        strategist.analyze,
        ticker=ticker,
        fiscal_year=fiscal_year,
        callbacks=callbacks,
    )

    return {"business_moat": moat_output}


async def financial_auditor_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 2: Audits multi-year financial statements, computes ratios, and checks red flags (Item 8)."""
    ticker = state["ticker"]
    fiscal_year = state["fiscal_year"]
    available_fiscal_years = state.get("available_fiscal_years")
    callbacks = state.get("callbacks")

    logger.info(f"[financial_auditor_node] Auditing {ticker} FY{fiscal_year} (available: {available_fiscal_years})")
    auditor = FinancialAuditorAgent()
    audit_output: FinancialAuditOutput = await asyncio.to_thread(
        auditor.audit,
        ticker=ticker,
        fiscal_year=fiscal_year,
        available_fiscal_years=available_fiscal_years,
        callbacks=callbacks,
    )

    return {"financial_audit": audit_output}


async def risk_analyst_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 3: Extracts material risk factors and structural threats (Item 1A)."""
    ticker = state["ticker"]
    fiscal_year = state["fiscal_year"]
    callbacks = state.get("callbacks")

    logger.info(f"[risk_analyst_node] Analyzing risks for {ticker} FY{fiscal_year}")
    risk_analyst = RiskAnalystAgent()
    risk_output: RiskAuditOutput = await asyncio.to_thread(
        risk_analyst.analyze,
        ticker=ticker,
        fiscal_year=fiscal_year,
        available_fiscal_years=state.get("available_fiscal_years"),
        callbacks=callbacks,
    )

    return {"risk_audit": risk_output}


TERMINAL_GROWTH_PATTERN = re.compile(
    r"(?:terminal growth|terminal rate|growth rate|perpetual growth)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)",
    re.IGNORECASE,
)


def _resolve_terminal_growth(user_query: str) -> float:
    """User-specified perpetual growth ('terminal growth 3%' or 'perpetual growth 0.02'), else the house default."""
    match = TERMINAL_GROWTH_PATTERN.search(user_query or "")
    if not match:
        return DEFAULT_TERMINAL_GROWTH
    raw = float(match.group(1))
    return raw / 100.0 if match.group(2) or raw > 1.0 else raw


async def forecasting_analyst_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 4: Forecasts revenue, margins and UFCFs (5 years, 10 when the growth fade is extended)
    from the audited baseline, Item 7 guidance and the Risk Analyst profile."""
    ticker = state["ticker"]
    fiscal_year = state["fiscal_year"]
    financial_audit = state.get("financial_audit")
    business_moat = state.get("business_moat")

    if financial_audit is None:
        raise ValueError(
            f"Cannot execute forecasting_analyst for {ticker}: missing required financial_audit."
        )

    callbacks = state.get("callbacks")
    logger.info(f"[forecasting_analyst_node] Building UFCF forecast for {ticker} FY{fiscal_year}")
    forecaster = ForecastingAnalystAgent()
    forecast_output: ForecastOutput = await asyncio.to_thread(
        forecaster.forecast,
        ticker=ticker,
        fiscal_year=fiscal_year,
        financial_audit=financial_audit,
        business_moat=business_moat,
        risk_audit=state.get("risk_audit"),
        terminal_growth=_resolve_terminal_growth(state.get("user_query", "")),
        callbacks=callbacks,
    )

    return {"forecast": forecast_output}


async def valuation_specialist_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 5: Computes CAPM WACC hurdle rate, 2-stage Gordon Growth DCF, and 5x5 sensitivity matrix."""
    ticker = state["ticker"]
    fiscal_year = state["fiscal_year"]
    financial_audit = state.get("financial_audit")
    forecast = state.get("forecast")
    user_query = state.get("user_query", "")
    callbacks = state.get("callbacks")

    if financial_audit is None:
        raise ValueError(
            f"Cannot execute valuation_specialist for {ticker}: missing required financial_audit."
        )

    # 1. Extract user-specified market parameters from the query
    beta_match = re.search(r"\bbeta\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\b", user_query, re.IGNORECASE)
    user_beta = float(beta_match.group(1)) if beta_match else None

    price_match = re.search(
        r"(?:price|trading at|share price)\s*[:=]?\s*\$?([0-9]+(?:\.[0-9]+)?)\b",
        user_query,
        re.IGNORECASE,
    )
    user_share_price = float(price_match.group(1)) if price_match else None

    terminal_growth = _resolve_terminal_growth(user_query)

    wacc_match = re.search(
        r"\b(?:wacc|discount rate|hurdle rate)\s*(?:of|is|at|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        user_query,
        re.IGNORECASE,
    )
    if not wacc_match:
        wacc_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:wacc|discount rate|hurdle rate)\b",
            user_query,
            re.IGNORECASE,
        )
    if wacc_match:
        w_raw = float(wacc_match.group(1))
        user_wacc = (w_raw / 100.0) if w_raw > 1.0 else w_raw
    else:
        user_wacc = None
        session_state = state.get("session_state") or {}
        if session_state.get("wacc_override") is not None:
            try:
                user_wacc = float(session_state["wacc_override"])
            except (ValueError, TypeError):
                user_wacc = None

    # 2. Resolve live market data (price, market cap, raw beta, 10Y Treasury) and the anchor balance-sheet date
    market_data, balance_sheet_date = await asyncio.gather(
        asyncio.to_thread(fetch_market_context, ticker),
        asyncio.to_thread(_balance_sheet_date, ticker, fiscal_year),
    )
    live_price = market_data.get("share_price")
    market = MarketInputs(
        share_price=user_share_price if user_share_price is not None else live_price,
        share_price_source=(
            "user_provided" if user_share_price is not None
            else ("yfinance_live" if live_price is not None else "unavailable")
        ),
        # A user price makes the live market cap inconsistent; weights then use price x valuation shares.
        market_cap=None if user_share_price is not None else market_data.get("market_cap"),
        beta=user_beta if user_beta is not None else market_data.get("beta"),
        beta_source="user_provided" if user_beta is not None else market_data.get("beta_source", "unavailable"),
        risk_free_rate=market_data.get("risk_free_rate", settings.DEFAULT_RISK_FREE_RATE),
        risk_free_rate_source=market_data.get("risk_free_rate_source", "house_default_config"),
        market_data_source=market_data.get("market_data_source", "fallback_offline"),
    )

    logger.info(
        f"[valuation_specialist_node] Valuing {ticker} (price=${market.share_price} [{market.share_price_source}], "
        f"beta={market.beta} [{market.beta_source}], rf={market.risk_free_rate} [{market.risk_free_rate_source}], "
        f"g={terminal_growth*100:.1f}%, balance_sheet_date={balance_sheet_date}, "
        f"user_wacc={f'{user_wacc*100:.2f}%' if user_wacc is not None else 'CAPM'})"
    )
    val_agent = ValuationSpecialistAgent()
    valuation_output: DCFValuationOutput = await asyncio.to_thread(
        val_agent.value,
        ticker=ticker,
        fiscal_year=fiscal_year,
        financial_audit=financial_audit,
        forecast=forecast,
        market=market,
        terminal_growth_rate=terminal_growth,
        wacc_override=user_wacc,
        valuation_date=date.today(),
        balance_sheet_date=balance_sheet_date,
        callbacks=callbacks,
    )

    return {"dcf_valuation": valuation_output}


def _balance_sheet_date(ticker: str, fiscal_year: int) -> Optional[str]:
    """Fiscal year-end (period_end_date) of the anchor 10-K, used to roll the DCF forward to today."""
    db = SessionLocal()
    try:
        row = (
            db.query(Document.period_end_date)
            .filter(
                Document.ticker == ticker.upper(),
                Document.fiscal_year == fiscal_year,
                Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
            )
            .first()
        )
        return row[0] if row and row[0] else None
    except Exception as e:
        logger.warning(f"Balance-sheet date lookup failed for {ticker} FY{fiscal_year}: {e}")
        return None
    finally:
        db.close()


async def lead_synthesizer_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 6: Synthesizes findings, builds 3-Pillar Thesis, and compiles publication report."""
    routing_plan = state.get("routing_plan")
    if routing_plan and routing_plan.needs_confirmation:
        return {
            "final_report": None,
            "sources": [],
            "error_message": routing_plan.confirmation_message,
        }

    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    fiscal_year = state["fiscal_year"]
    year_substituted = routing_plan.year_substituted if routing_plan else False
    user_query = state.get("user_query")
    query_type = state.get("query_type", "full_10k_report")
    callbacks = state.get("callbacks")

    logger.info(
        f"[lead_synthesizer_node] Compiling report for {company_name} ({ticker}) [query_type={query_type}]"
    )
    synthesizer = LeadSynthesizerAgent()
    report: Final10KResearchReport = await asyncio.to_thread(
        synthesizer.synthesize,
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        business_moat=state.get("business_moat"),
        financial_audit=state.get("financial_audit"),
        forecast=state.get("forecast"),
        dcf_valuation=state.get("dcf_valuation"),
        risk_audit=state.get("risk_audit"),
        year_substituted=year_substituted,
        available_fiscal_years=state.get("available_fiscal_years"),
        user_query=user_query,
        query_type=query_type,
        callbacks=callbacks,
    )

    return {
        "final_report": report,
        "sources": report.all_citations,
    }


# ==============================================================================
# 2. Conditional Routing Predicates
# ==============================================================================
def route_from_supervisor(state: EquityResearchState) -> List[str]:
    """Branches execution path based on resolved QueryType."""
    qtype = state.get("query_type", "full_10k_report")
    if qtype == "conversational":
        return [END]
    elif qtype == "business_moat_only":
        return ["business_strategist"]
    elif qtype == "financial_audit_only":
        return ["financial_auditor"]
    elif qtype == "risk_factors_only":
        return ["risk_analyst"]
    elif qtype == "dcf_valuation_only":
        return ["financial_auditor"]
    else:  # "full_10k_report"
        return ["business_strategist", "financial_auditor", "risk_analyst"]


def route_from_business_strategist(state: EquityResearchState) -> str:
    """Routes business strategist output to synthesis (if standalone) or forecasting."""
    return (
        "lead_synthesizer"
        if state.get("query_type") == "business_moat_only"
        else "forecasting_analyst"
    )


def route_from_financial_auditor(state: EquityResearchState) -> str:
    """Routes financial auditor output to synthesis (if standalone) or forecasting."""
    return (
        "lead_synthesizer"
        if state.get("query_type") == "financial_audit_only"
        else "forecasting_analyst"
    )


def route_from_risk_analyst(state: EquityResearchState) -> str:
    """Routes risk analyst output to synthesis (if standalone) or forecasting."""
    return (
        "lead_synthesizer"
        if state.get("query_type") == "risk_factors_only"
        else "forecasting_analyst"
    )


# ==============================================================================
# 3. LangGraph StateGraph Assembly
# ==============================================================================
def build_equity_research_graph() -> CompiledStateGraph:
    """Constructs and compiles the complete 6-stage LangGraph multi-agent state machine."""
    builder = StateGraph(EquityResearchState)

    # 1. Register all nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("business_strategist", business_strategist_node)
    builder.add_node("financial_auditor", financial_auditor_node)
    builder.add_node("risk_analyst", risk_analyst_node)
    builder.add_node("forecasting_analyst", forecasting_analyst_node)
    builder.add_node("valuation_specialist", valuation_specialist_node)
    builder.add_node("lead_synthesizer", lead_synthesizer_node)

    # 2. Lead Supervisor Entry
    builder.add_edge(START, "supervisor")

    # 3. Fan-out conditional routing from Supervisor
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        [
            "business_strategist",
            "financial_auditor",
            "risk_analyst",
            "lead_synthesizer",
            END,
        ],
    )

    # 4. Phase 1 Qualitative & Statement Auditing -> Forecasting Join
    builder.add_conditional_edges(
        "business_strategist",
        route_from_business_strategist,
        ["lead_synthesizer", "forecasting_analyst"],
    )
    builder.add_conditional_edges(
        "financial_auditor",
        route_from_financial_auditor,
        ["lead_synthesizer", "forecasting_analyst"],
    )
    builder.add_conditional_edges(
        "risk_analyst",
        route_from_risk_analyst,
        ["lead_synthesizer", "forecasting_analyst"],
    )

    # 5. Sequential Valuation & Synthesis Chain
    builder.add_edge("forecasting_analyst", "valuation_specialist")
    builder.add_edge("valuation_specialist", "lead_synthesizer")
    builder.add_edge("lead_synthesizer", END)

    return builder.compile()


# ==============================================================================
# 4. MultiAgentOrchestrator (Registered BaseAgent)
# ==============================================================================
@AgentRegistry.register("multi_agent")
@AgentRegistry.register("orchestrator")
class MultiAgentOrchestrator(BaseAgent):
    """Institutional equity research coordinator wrapping the LangGraph state machine."""

    def __init__(self, recursion_limit: int = 50):
        self.recursion_limit = recursion_limit
        self._cached_graph: Optional[CompiledStateGraph] = None

    def get_graph(self) -> CompiledStateGraph:
        """Cache-first resolver for the compiled LangGraph execution graph."""
        if self._cached_graph is None:
            self._cached_graph = build_equity_research_graph()
        return self._cached_graph

    @staticmethod
    def _forecast_status_message(fc: Optional[ForecastOutput]) -> str:
        if fc is None:
            return "Forecast unavailable"
        source = "MD&A-informed" if fc.guidance_source == "mdna_informed" else "historical fade"
        return (
            f"Built {fc.forecast_horizon_years}-year UFCF forecast ({source}, revenue CAGR "
            f"{fc.revenue_cagr_pct:.1f}%, {len(fc.scenarios)} scenarios)"
        )

    @staticmethod
    def _valuation_status_message(val: Optional[DCFValuationOutput]) -> str:
        if val is None:
            return "Valuation unavailable"
        message = (
            f"Discounted at {val.wacc_audit.wacc_pct:.2f}% "
            f"({'user override' if val.wacc_audit.wacc_source == 'user_override' else 'CAPM WACC'}); "
            f"base-case DCF fair value ${val.implied_fair_value_per_share:.2f}/share"
        )
        if val.bear_fair_value is not None and val.bull_fair_value is not None:
            message += f" (bear ${val.bear_fair_value:.2f} - bull ${val.bull_fair_value:.2f})"
        return message

    @staticmethod
    def _risk_status_message(risk: Optional[RiskAuditOutput]) -> str:
        """SSE status line summarizing the verified risk inventory."""
        if risk is None:
            return "Analyzed Item 1A risks & threats (no risk output)"
        if risk.data_quality.coverage_mode == "not_provided":
            return "Analyzed Item 1A risks & threats (filing provides no risk factors; nothing rated)"
        severe = sum(1 for r in risk.identified_risks if r.severity == "Severe")
        coverage = risk.data_quality.coverage_mode.replace("_", "-")
        if risk.data_quality.section_located_by == "heading_boundary":
            coverage += " (located by headings)"
        if risk.disclosure_changes:
            new_cnt = sum(1 for r in risk.identified_risks if r.disclosure_change in ("New", "Expanded"))
            change = f"{new_cnt} new/expanded vs FY{risk.disclosure_changes.prior_fiscal_year}"
        else:
            change = "no prior-year baseline"
        return (
            f"Analyzed Item 1A risks & threats ({len(risk.identified_risks)} verified risks, {severe} severe; "
            f"{coverage} read; {change})"
        )

    async def arun(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> EquityResearchState:
        """Executes the multi-agent graph asynchronously and returns the populated final state."""
        graph = self.get_graph()
        initial_state: EquityResearchState = {
            "user_query": user_query,
            "ticker": ticker.upper() if ticker else "",
            "fiscal_year": fiscal_year or 0,
            "messages": messages,
            "session_state": session_state,
            "callbacks": callbacks,
        }

        config: Dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if callbacks:
            config["callbacks"] = callbacks

        final_state = await graph.ainvoke(
            initial_state,
            config=config,
        )
        return final_state

    async def astream_run(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Executes the graph while streaming real-time intermediate node progress milestones."""
        resolved_ticker = ticker.upper() if ticker else ""

        graph = self.get_graph()
        initial_state: EquityResearchState = {
            "user_query": user_query,
            "ticker": resolved_ticker,
            "fiscal_year": fiscal_year or 0,
            "messages": messages,
            "session_state": session_state,
            "callbacks": callbacks,
        }

        config: Dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if callbacks:
            config["callbacks"] = callbacks

        yield {
            "type": "status",
            "node": "supervisor",
            "message": "Analyzing research inquiry & context...",
        }


        final_state: Dict[str, Any] = {}

        try:
            async for chunk in graph.astream(
                initial_state,
                config=config,
            ):
                for node_name, node_output in chunk.items():
                    final_state.update(node_output)

                    if node_name == "supervisor":
                        plan: Optional[RoutingPlan] = node_output.get("routing_plan")
                        t = node_output.get("ticker", "Target")
                        y = node_output.get("fiscal_year", "")
                        q = node_output.get("query_type", "full_10k_report")
                        if q != "conversational":
                            sub_note = " (substituted year)" if (plan and plan.year_substituted) else ""
                            yield {
                                "type": "status",
                                "node": "supervisor",
                                "message": f"Resolved filing: {t} FY{y}{sub_note} | Route: {q}",
                                "details": {
                                    "ticker": t,
                                    "fiscal_year": y,
                                    "query_type": q,
                                    "year_substituted": plan.year_substituted if plan else False,
                                },
                            }
                    elif node_name == "business_strategist":
                        moat: Optional[BusinessMoatOutput] = node_output.get("business_moat")
                        m_type = moat.economic_moat_type if moat else "Assessed"
                        yield {
                            "type": "status",
                            "node": "business_strategist",
                            "message": f"Completed economic moat analysis (Moat: {m_type})",
                        }
                    elif node_name == "financial_auditor":
                        audit: Optional[FinancialAuditOutput] = node_output.get("financial_audit")
                        findings = audit.forensic_findings if audit else []
                        severe_cnt = sum(1 for f in findings if f.severity in ("high", "medium"))
                        mode = audit.data_quality.extraction_mode if audit else "n/a"
                        yield {
                            "type": "status",
                            "node": "financial_auditor",
                            "message": (
                                f"Audited financial statements & ratios ({severe_cnt} high/medium forensic "
                                f"findings, {mode} extraction)"
                            ),
                        }
                    elif node_name == "risk_analyst":
                        risk: Optional[RiskAuditOutput] = node_output.get("risk_audit")
                        yield {
                            "type": "status",
                            "node": "risk_analyst",
                            "message": self._risk_status_message(risk),
                        }
                    elif node_name == "forecasting_analyst":
                        fc: Optional[ForecastOutput] = node_output.get("forecast")
                        yield {
                            "type": "status",
                            "node": "forecasting_analyst",
                            "message": self._forecast_status_message(fc),
                        }
                    elif node_name == "valuation_specialist":
                        val: Optional[DCFValuationOutput] = node_output.get("dcf_valuation")
                        yield {
                            "type": "status",
                            "node": "valuation_specialist",
                            "message": self._valuation_status_message(val),
                        }
                    elif node_name == "lead_synthesizer":
                        report: Optional[Final10KResearchReport] = node_output.get("final_report")
                        yield {
                            "type": "status",
                            "node": "lead_synthesizer",
                            "message": "Synthesized 3-Pillar Thesis and compiled final publication report.",
                        }
        except ValueError as e:
            logger.warning(f"astream_run caught catalog resolution error: {e}")
            yield {
                "type": "result",
                "response": (
                    f"### ℹ️ Research Terminal Guidance\n\n"
                    f"{str(e)}\n\n"
                    f"**Suggested Inquiries:**\n"
                    f"- `Analyze Apple` (triggers full 6-agent equity research report)\n"
                    f"- `Analyze Tesla` or `Analyze TSLA`\n"
                    f"- `Analyze NVIDIA` or `Analyze NVDA`\n\n"
                    f"*Tip: You can also attach any SEC 10-K filing using the **Attach .htm** button.*"
                ),
                "sources": [],
            }
            return

        report_obj = final_state.get("final_report")
        conv_resp = final_state.get("conversational_response")
        updated_state = final_state.get("updated_session_state")
        if conv_resp:
            yield {
                "type": "result",
                "response": conv_resp,
                "sources": final_state.get("sources", []),
                "updated_session_state": updated_state,
            }
        elif isinstance(report_obj, Final10KResearchReport):
            yield {
                "type": "result",
                "response": report_obj.full_markdown_report,
                "sources": report_obj.all_citations,
                "final_report": report_obj.model_dump(),
                "updated_session_state": updated_state,
            }
        else:
            yield {
                "type": "result",
                "response": final_state.get("error_message") or "Pipeline execution finished.",
                "sources": final_state.get("sources", []),
                "updated_session_state": updated_state,
            }

    def run(
        self,
        messages: List[Dict[str, str]],
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> AgentOutput:
        """Executes the agent synchronously conforming to BaseAgent interface."""
        query = messages[-1]["content"] if messages else ""

        # Safely execute async coroutine inside sync context without blocking or loop collisions
        extracted_ticker = None
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    state = pool.submit(
                        asyncio.run,
                        self.arun(
                            user_query=query,
                            ticker=extracted_ticker,
                            messages=messages,
                            session_state=session_state,
                            callbacks=callbacks,
                        ),
                    ).result()
            else:
                state = asyncio.run(
                    self.arun(
                        user_query=query,
                        ticker=extracted_ticker,
                        messages=messages,
                        session_state=session_state,
                        callbacks=callbacks,
                    )
                )

            conv_resp = state.get("conversational_response")
            final_report = state.get("final_report")
            updated_state = state.get("updated_session_state")
            if conv_resp:
                return AgentOutput(
                    content=conv_resp,
                    sources=state.get("sources", []),
                    updated_session_state=updated_state,
                )
            if isinstance(final_report, Final10KResearchReport):
                return AgentOutput(
                    content=final_report.full_markdown_report,
                    sources=final_report.all_citations,
                    updated_session_state=updated_state,
                )

            error = state.get("error_message") or "Equity research execution completed without final report."
            return AgentOutput(
                content=error,
                sources=state.get("sources", []),
                updated_session_state=updated_state,
            )

        except ValueError as e:
            logger.warning(f"MultiAgentOrchestrator.run caught resolution error: {e}")
            return AgentOutput(
                content=(
                    f"### ℹ️ Research Terminal Guidance\n\n"
                    f"{str(e)}\n\n"
                    f"**Suggested Inquiries:**\n"
                    f"- `Analyze Apple` (triggers full 6-agent equity research report)\n"
                    f"- `Analyze Tesla` or `Analyze TSLA`\n"
                    f"- `Analyze NVIDIA` or `Analyze NVDA`\n\n"
                    f"*Tip: You can also attach any SEC 10-K filing using the **Attach .htm** button.*"
                ),
                sources=[],
            )
