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
- Failure isolation: a failed specialist is recorded in `pipeline_notes`, its section is omitted, and the run
  continues; only a run with no research output at all fails.
- Milestone status event streaming for real-time UI updates.
"""

import asyncio
import concurrent.futures
import logging
from datetime import date
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.base import AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.business_strategist import BusinessStrategistAgent
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
from app.agents.specialized.lead_synthesizer import LeadSynthesizerAgent
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
def _failure_note(agent_label: str, exc: BaseException) -> str:
    detail = " ".join(str(exc).split())[:240] or type(exc).__name__
    return f"{agent_label} failed ({type(exc).__name__}: {detail}); its section is omitted from this report."


def _overrides(state: EquityResearchState) -> Dict[str, Any]:
    plan: Optional[RoutingPlan] = state.get("routing_plan")
    return dict(plan.valuation_overrides) if plan else {}


def _terminal_growth(state: EquityResearchState) -> float:
    """User-specified perpetual growth (range-checked by the supervisor), else the house default."""
    value = _overrides(state).get("terminal_growth")
    return float(value) if value is not None else DEFAULT_TERMINAL_GROWTH


async def supervisor_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 0: Triages user inquiry, resolves filing catalog, and emits RoutingPlan."""
    supervisor = SupervisorAgent()
    routing_plan: RoutingPlan = await asyncio.to_thread(
        supervisor.route,
        user_query=state.get("user_query", ""),
        ticker=state.get("ticker"),
        fiscal_year=state.get("fiscal_year"),
        session_state=state.get("session_state"),
        messages=state.get("messages"),
        callbacks=state.get("callbacks"),
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
        "pipeline_notes": list(routing_plan.parameter_notes),
    }


async def business_strategist_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 1: Evaluates business model, product segments, and economic moat (Item 1). Failure is non-fatal."""
    ticker, fiscal_year = state["ticker"], state["fiscal_year"]
    logger.info(f"[business_strategist_node] Analyzing {ticker} FY{fiscal_year}")
    try:
        moat_output: BusinessMoatOutput = await asyncio.to_thread(
            BusinessStrategistAgent().analyze,
            ticker=ticker,
            fiscal_year=fiscal_year,
            callbacks=state.get("callbacks"),
        )
    except Exception as e:
        logger.exception(f"[business_strategist_node] {ticker} FY{fiscal_year} failed")
        return {"business_moat": None, "pipeline_notes": [_failure_note("Business Strategist", e)]}
    return {"business_moat": moat_output}


async def financial_auditor_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 2: Audits multi-year financial statements (Item 8). The agent itself falls back to its deterministic
    baseline when the LLM step fails; a failure here means no statements could be audited at all."""
    ticker, fiscal_year = state["ticker"], state["fiscal_year"]
    available_fiscal_years = state.get("available_fiscal_years")
    logger.info(f"[financial_auditor_node] Auditing {ticker} FY{fiscal_year} (available: {available_fiscal_years})")
    try:
        audit_output: FinancialAuditOutput = await asyncio.to_thread(
            FinancialAuditorAgent().audit,
            ticker=ticker,
            fiscal_year=fiscal_year,
            available_fiscal_years=available_fiscal_years,
            callbacks=state.get("callbacks"),
        )
    except Exception as e:
        logger.exception(f"[financial_auditor_node] {ticker} FY{fiscal_year} failed")
        return {"financial_audit": None, "pipeline_notes": [_failure_note("Financial Auditor", e)]}
    return {"financial_audit": audit_output}


async def risk_analyst_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 3: Extracts material risk factors and structural threats (Item 1A). Failure is non-fatal."""
    ticker, fiscal_year = state["ticker"], state["fiscal_year"]
    logger.info(f"[risk_analyst_node] Analyzing risks for {ticker} FY{fiscal_year}")
    try:
        risk_output: RiskAuditOutput = await asyncio.to_thread(
            RiskAnalystAgent().analyze,
            ticker=ticker,
            fiscal_year=fiscal_year,
            available_fiscal_years=state.get("available_fiscal_years"),
            callbacks=state.get("callbacks"),
        )
    except Exception as e:
        logger.exception(f"[risk_analyst_node] {ticker} FY{fiscal_year} failed")
        return {"risk_audit": None, "pipeline_notes": [_failure_note("Risk Analyst", e)]}
    return {"risk_audit": risk_output}


async def forecasting_analyst_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 4: Forecasts revenue, margins and UFCFs from the audited baseline, Item 7 guidance and the
    Risk Analyst profile. Skipped (with a note) when no audited financials are available."""
    ticker, fiscal_year = state["ticker"], state["fiscal_year"]
    financial_audit = state.get("financial_audit")
    if financial_audit is None:
        return {"forecast": None, "pipeline_notes": ["Forecast skipped: no audited financial statements were available."]}

    logger.info(f"[forecasting_analyst_node] Building UFCF forecast for {ticker} FY{fiscal_year}")
    try:
        forecast_output: ForecastOutput = await asyncio.to_thread(
            ForecastingAnalystAgent().forecast,
            ticker=ticker,
            fiscal_year=fiscal_year,
            financial_audit=financial_audit,
            business_moat=state.get("business_moat"),
            risk_audit=state.get("risk_audit"),
            terminal_growth=_terminal_growth(state),
            callbacks=state.get("callbacks"),
        )
    except Exception as e:
        logger.exception(f"[forecasting_analyst_node] {ticker} FY{fiscal_year} failed")
        return {"forecast": None, "pipeline_notes": [_failure_note("Forecasting Analyst", e)]}
    return {"forecast": forecast_output}


async def valuation_specialist_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 5: CAPM WACC, 2-stage DCF, scenarios and sensitivity. Skipped (with a note) without audited
    financials; a valuation failure is non-fatal."""
    ticker, fiscal_year = state["ticker"], state["fiscal_year"]
    financial_audit = state.get("financial_audit")
    if financial_audit is None:
        return {"dcf_valuation": None, "pipeline_notes": ["Valuation skipped: no audited financial statements were available."]}

    overrides = _overrides(state)
    user_beta, user_share_price, user_wacc = overrides.get("beta"), overrides.get("share_price"), overrides.get("wacc")
    terminal_growth = _terminal_growth(state)

    # Live market data (price, market cap, raw beta, 10Y Treasury) and the anchor balance-sheet date
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

    wacc_label = f"{user_wacc * 100:.2f}%" if user_wacc is not None else "CAPM"
    logger.info(
        f"[valuation_specialist_node] Valuing {ticker} (price=${market.share_price} [{market.share_price_source}], "
        f"beta={market.beta} [{market.beta_source}], rf={market.risk_free_rate} [{market.risk_free_rate_source}], "
        f"g={terminal_growth * 100:.1f}%, balance_sheet_date={balance_sheet_date}, user_wacc={wacc_label})"
    )
    try:
        valuation_output: DCFValuationOutput = await asyncio.to_thread(
            ValuationSpecialistAgent().value,
            ticker=ticker,
            fiscal_year=fiscal_year,
            financial_audit=financial_audit,
            forecast=state.get("forecast"),
            market=market,
            terminal_growth_rate=terminal_growth,
            wacc_override=user_wacc,
            valuation_date=date.today(),
            balance_sheet_date=balance_sheet_date,
            callbacks=state.get("callbacks"),
        )
    except Exception as e:
        logger.exception(f"[valuation_specialist_node] {ticker} FY{fiscal_year} failed")
        return {"dcf_valuation": None, "pipeline_notes": [_failure_note("Valuation Specialist", e)]}
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
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    fiscal_year = state["fiscal_year"]
    query_type = state.get("query_type", "full_10k_report")
    notes = list(state.get("pipeline_notes") or [])

    if all(state.get(k) is None for k in ("business_moat", "financial_audit", "risk_audit")):
        raise RuntimeError(f"No research output could be produced for {ticker} FY{fiscal_year}. " + " ".join(notes))

    logger.info(
        f"[lead_synthesizer_node] Compiling report for {company_name} ({ticker}) [query_type={query_type}]"
    )
    report: Final10KResearchReport = await asyncio.to_thread(
        LeadSynthesizerAgent().synthesize,
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        business_moat=state.get("business_moat"),
        financial_audit=state.get("financial_audit"),
        forecast=state.get("forecast"),
        dcf_valuation=state.get("dcf_valuation"),
        risk_audit=state.get("risk_audit"),
        year_substituted=routing_plan.year_substituted if routing_plan else False,
        available_fiscal_years=state.get("available_fiscal_years"),
        user_query=state.get("user_query"),
        query_type=query_type,
        pipeline_notes=notes,
        callbacks=state.get("callbacks"),
    )

    return {
        "final_report": report,
        "sources": [s.model_dump() for s in report.sources],
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
            async for chunk in graph.astream(initial_state, config=config):
                for node_name, node_output in chunk.items():
                    final_state.update(node_output)
                    message, details = self._node_status(node_name, node_output)
                    if message:
                        event: Dict[str, Any] = {"type": "status", "node": node_name, "message": message}
                        if details:
                            event["details"] = details
                        yield event
                    for note in node_output.get("pipeline_notes") or []:
                        yield {"type": "status", "node": node_name, "message": note, "details": {"severity": "warning"}}
        except Exception as e:
            # Stream an error event instead of tearing the SSE connection down; the chat service persists it.
            logger.exception("astream_run: research pipeline failed")
            yield {"type": "error", "message": f"Research pipeline failed: {e}"}
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
                "sources": [s.model_dump() for s in report_obj.sources],
                "final_report": report_obj.model_dump(),
                "updated_session_state": updated_state,
            }
        else:
            yield {
                "type": "error",
                "message": final_state.get("error_message") or "The research pipeline finished without a report.",
            }

    def _node_status(self, node_name: str, node_output: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Milestone message (and optional details) for a finished node; None when nothing should be shown."""
        if node_name == "supervisor":
            plan: Optional[RoutingPlan] = node_output.get("routing_plan")
            if node_output.get("query_type") == "conversational" or plan is None:
                return None, None
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
        if node_name == "business_strategist":
            moat: Optional[BusinessMoatOutput] = node_output.get("business_moat")
            if moat is None:
                return "Economic moat analysis unavailable", None
            return f"Completed economic moat analysis (Moat: {moat.economic_moat_type})", None
        if node_name == "financial_auditor":
            audit: Optional[FinancialAuditOutput] = node_output.get("financial_audit")
            if audit is None:
                return "Financial statement audit unavailable", None
            severe_cnt = sum(1 for f in audit.forensic_findings if f.severity in ("high", "medium"))
            return (
                f"Audited financial statements & ratios ({severe_cnt} high/medium forensic findings, "
                f"{audit.data_quality.extraction_mode} extraction)",
                None,
            )
        if node_name == "risk_analyst":
            return self._risk_status_message(node_output.get("risk_audit")), None
        if node_name == "forecasting_analyst":
            return self._forecast_status_message(node_output.get("forecast")), None
        if node_name == "valuation_specialist":
            return self._valuation_status_message(node_output.get("dcf_valuation")), None
        if node_name == "lead_synthesizer":
            return "Synthesized 3-Pillar Thesis and compiled final publication report.", None
        return None, None

    def run(
        self,
        messages: List[Dict[str, str]],
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> AgentOutput:
        """Executes the agent synchronously conforming to BaseAgent interface. Failures propagate to the caller."""
        query = messages[-1]["content"] if messages else ""
        coroutine = self.arun(user_query=query, messages=messages, session_state=session_state, callbacks=callbacks)

        # Safely execute the coroutine inside a sync context without loop collisions
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = pool.submit(asyncio.run, coroutine).result()
        else:
            state = asyncio.run(coroutine)

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
                sources=[s.model_dump() for s in final_report.sources],
                updated_session_state=updated_state,
            )
        raise RuntimeError(state.get("error_message") or "The research pipeline finished without a report.")
