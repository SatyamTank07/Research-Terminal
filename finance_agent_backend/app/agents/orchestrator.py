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
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.base import AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.business_strategist import BusinessStrategistAgent
from app.agents.specialized.conversational.agent_conversational import ConversationalAnalystAgent
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
from app.agents.tools.market_data_tools import fetch_market_context
from app.database import SessionLocal
from app.models import Document

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
        f"route={routing_plan.query_type} (needs_confirmation={routing_plan.needs_confirmation})"
    )

    return {
        "ticker": routing_plan.ticker,
        "company_name": routing_plan.company_name or routing_plan.ticker,
        "fiscal_year": routing_plan.fiscal_year,
        "document_id": routing_plan.document_id,
        "query_type": routing_plan.query_type,
        "routing_plan": routing_plan,
        "error_message": routing_plan.confirmation_message if routing_plan.needs_confirmation else None,
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
    callbacks = state.get("callbacks")

    logger.info(f"[financial_auditor_node] Auditing {ticker} FY{fiscal_year}")
    auditor = FinancialAuditorAgent()
    audit_output: FinancialAuditOutput = await asyncio.to_thread(
        auditor.audit,
        ticker=ticker,
        fiscal_year=fiscal_year,
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
        callbacks=callbacks,
    )

    return {"risk_audit": risk_output}


async def forecasting_analyst_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node 4: Forecasts 5-year revenue, margins, and UFCFs using audit history and segment insights."""
    ticker = state["ticker"]
    fiscal_year = state["fiscal_year"]
    financial_audit = state.get("financial_audit")
    business_moat = state.get("business_moat")

    if financial_audit is None:
        raise ValueError(
            f"Cannot execute forecasting_analyst for {ticker}: missing required financial_audit."
        )

    callbacks = state.get("callbacks")
    logger.info(f"[forecasting_analyst_node] Building 5-yr UFCF schedule for {ticker} FY{fiscal_year}")
    forecaster = ForecastingAnalystAgent()
    forecast_output: ForecastOutput = await asyncio.to_thread(
        forecaster.forecast,
        ticker=ticker,
        fiscal_year=fiscal_year,
        financial_audit=financial_audit,
        business_moat=business_moat,
        horizon_years=5,
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

    # 1. Resolve projected cash flows from forecaster (or base FCFs if unavailable)
    if forecast and forecast.projected_fcfs:
        projected_fcfs = forecast.projected_fcfs
    else:
        # Fallback to historical latest FCF if forecasting was skipped
        base_fcf = financial_audit.balance_sheet.diluted_shares_outstanding * 5.0
        if financial_audit.multi_year_history:
            base_fcf = financial_audit.multi_year_history[-1].free_cash_flow
        projected_fcfs = [base_fcf * (1.05 ** i) for i in range(1, 6)]

    # 2. Extract or default market parameters from user query
    beta_match = re.search(r"\bbeta\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\b", user_query, re.IGNORECASE)
    user_beta = float(beta_match.group(1)) if beta_match else None

    price_match = re.search(
        r"(?:price|trading at|share price)\s*[:=]?\s*\$?([0-9]+(?:\.[0-9]+)?)\b",
        user_query,
        re.IGNORECASE,
    )
    user_share_price = float(price_match.group(1)) if price_match else None

    growth_match = re.search(
        r"(?:terminal growth|terminal rate|growth rate|perpetual growth)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        user_query,
        re.IGNORECASE,
    )
    if growth_match:
        g_raw = float(growth_match.group(1))
        terminal_growth = (g_raw / 100.0) if g_raw > 1.0 else g_raw
    else:
        terminal_growth = 0.025

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

    # 3. Resolve live market data (share_price, market_cap, beta) via yfinance with graceful fallbacks
    market_data = await asyncio.to_thread(fetch_market_context, ticker)
    share_price = user_share_price if user_share_price is not None else market_data.get("share_price")
    market_cap = market_data.get("market_cap")
    beta = user_beta if user_beta is not None else market_data.get("beta", 1.0)

    # Fallback to book stockholders_equity if both market_cap and share_price are missing
    if market_cap is None and share_price is None:
        book_equity = getattr(financial_audit.balance_sheet, "stockholders_equity", None)
        if book_equity and book_equity > 0:
            market_cap = float(book_equity)
        else:
            market_cap = max(float(financial_audit.balance_sheet.total_debt) * 4.0, 10000.0)

    logger.info(
        f"[valuation_specialist_node] Valuing {ticker} (price=${share_price}, "
        f"market_cap=${market_cap}M, beta={beta:.2f}, g={terminal_growth*100:.1f}%, "
        f"user_wacc={f'{user_wacc*100:.2f}%' if user_wacc is not None else 'CAPM'})"
    )
    val_agent = ValuationSpecialistAgent()
    valuation_output: DCFValuationOutput = await asyncio.to_thread(
        val_agent.value,
        ticker=ticker,
        fiscal_year=fiscal_year,
        financial_audit=financial_audit,
        projected_fcfs=projected_fcfs,
        beta=beta,
        share_price=share_price,
        market_cap=market_cap,
        terminal_growth_rate=terminal_growth,
        wacc_override=user_wacc,
        callbacks=callbacks,
    )

    return {"dcf_valuation": valuation_output}


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
        user_query=user_query,
        query_type=query_type,
        callbacks=callbacks,
    )

    return {
        "final_report": report,
        "sources": report.all_citations,
    }


async def conversational_node(state: EquityResearchState) -> Dict[str, Any]:
    """Node: Delegates conversational inquiries, capabilities, and follow-ups to ConversationalAnalystAgent."""
    user_query = state.get("user_query", "")
    messages = state.get("messages")
    session_state = state.get("session_state")
    callbacks = state.get("callbacks")

    agent = ConversationalAnalystAgent()
    answer = await agent.arespond(
        user_query=user_query,
        messages=messages,
        session_state=session_state,
        callbacks=callbacks,
    )

    return {
        "conversational_response": answer,
        "error_message": None,
        "updated_session_state": session_state,
    }


# ==============================================================================
# 2. Conditional Routing Predicates
# ==============================================================================
def route_from_supervisor(state: EquityResearchState) -> List[str]:
    """Branches execution path based on resolved QueryType."""
    plan = state.get("routing_plan")
    if plan and plan.needs_confirmation:
        return ["lead_synthesizer"]

    qtype = state.get("query_type", "full_10k_report")
    if qtype == "conversational":
        return ["conversational_analyst"]
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
    builder.add_node("conversational_analyst", conversational_node)
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
            "conversational_analyst",
        ],
    )

    # Conversational path completes directly
    builder.add_edge("conversational_analyst", END)

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
        # Scan previous messages backwards if current query does not explicitly specify a ticker
        extracted_ticker = ticker.upper() if ticker else ""

        if not extracted_ticker and messages and len(messages) > 1:
            from app.database import SessionLocal
            db = SessionLocal()
            try:
                sup = SupervisorAgent()
                for m in reversed(messages[:-1]):
                    candidate = sup._extract_ticker_from_query(m.get("content", ""), db)
                    if candidate:
                        extracted_ticker = candidate
                        break
            except Exception as e:
                logger.debug(f"Stream history ticker scan exception: {e}")
            finally:
                db.close()

        graph = self.get_graph()
        initial_state: EquityResearchState = {
            "user_query": user_query,
            "ticker": extracted_ticker,
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
                        if plan and plan.needs_confirmation:
                            yield {
                                "type": "status",
                                "node": "supervisor",
                                "message": f"Requested year not found. Prompting user to confirm latest FY{plan.suggested_fiscal_year}.",
                                "details": {
                                    "ticker": t,
                                    "needs_confirmation": True,
                                    "suggested_fiscal_year": plan.suggested_fiscal_year,
                                },
                            }
                        elif q == "conversational":
                            yield {
                                "type": "status",
                                "node": "supervisor",
                                "message": "Directing inquiry to Research Assistant...",
                                "details": {
                                    "query_type": "conversational",
                                },
                            }
                        else:
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
                    elif node_name == "conversational_analyst":
                        yield {
                            "type": "status",
                            "node": "conversational_analyst",
                            "message": "Formulating research response...",
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
                        flags_cnt = len(audit.forensic_red_flags) if audit else 0
                        yield {
                            "type": "status",
                            "node": "financial_auditor",
                            "message": f"Audited financial statements & ratios ({flags_cnt} red flags evaluated)",
                        }
                    elif node_name == "risk_analyst":
                        risk: Optional[RiskAuditOutput] = node_output.get("risk_audit")
                        top_cnt = len(risk.identified_risks) if risk else 0
                        yield {
                            "type": "status",
                            "node": "risk_analyst",
                            "message": f"Analyzed Item 1A risks & threats ({top_cnt} key risks identified)",
                        }
                    elif node_name == "forecasting_analyst":
                        fc: Optional[ForecastOutput] = node_output.get("forecast")
                        cagr = fc.revenue_cagr_pct if fc else 0.0
                        yield {
                            "type": "status",
                            "node": "forecasting_analyst",
                            "message": f"Constructed 5-year UFCF projection schedule (Revenue CAGR: {cagr:.1f}%)",
                        }
                    elif node_name == "valuation_specialist":
                        val: Optional[DCFValuationOutput] = node_output.get("dcf_valuation")
                        fv = val.implied_fair_value_per_share if val else 0.0
                        w = val.wacc_audit.wacc_pct if val else 0.0
                        yield {
                            "type": "status",
                            "node": "valuation_specialist",
                            "message": f"Derived CAPM WACC ({w:.2f}%) and calculated DCF Fair Value (${fv:.2f}/share)",
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

        # Scan previous messages backwards if current query does not explicitly specify a ticker
        extracted_ticker = None
        if len(messages) > 1:
            from app.database import SessionLocal
            db = SessionLocal()
            try:
                sup = SupervisorAgent()
                for m in reversed(messages[:-1]):
                    candidate = sup._extract_ticker_from_query(m.get("content", ""), db)
                    if candidate:
                        extracted_ticker = candidate
                        break
            except Exception as e:
                logger.debug(f"History ticker scan exception: {e}")
            finally:
                db.close()

        # Safely execute async coroutine inside sync context without blocking or loop collisions
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
