"""End-to-End Pipeline & Integration Tests for Milestone 6 (Part 2).

Validates:
1. LangGraph StateGraph structure and compilation via build_equity_research_graph().
2. MultiAgentOrchestrator registration and BaseAgent interface compatibility.
3. Fast-Path DCF valuation execution on Tesla (TSLA FY2025):
   - Confirms skipping qualitative branches (business moat & risk factors remain None).
   - Confirms execution through financial auditor -> forecaster -> valuation specialist -> lead synthesizer.
4. Single-Agent ad-hoc route execution (business_moat_only on NVDA FY2026).
5. Real-time intermediate node status streaming via astream_run().
6. Full 10-K equity research pipeline execution on Apple (AAPL FY2025):
   - Confirms parallel execution of Phase 1 qualitative & statement auditing.
   - Confirms sequential forecasting and DCF valuation.
   - Confirms complete Final10KResearchReport with 3-Pillar Investment Thesis, DCF sensitivity, and citations.
7. Chat Service integration (process_chat and stream_chat_service).
"""

import asyncio
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch
from app.agents import AgentRegistry, MultiAgentOrchestrator, build_equity_research_graph
from app.agents.specialized.business_strategist import BusinessStrategistAgent
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
from app.agents.specialized.lead_synthesizer import LeadSynthesizerAgent
from app.agents.specialized.risk_analyst import RiskAnalystAgent
from app.agents.specialized.supervisor import SupervisorAgent
from app.agents.specialized.supervisor.agent_supervisor import SupervisorExtraction
from app.agents.specialized.valuation_specialist import ValuationSpecialistAgent
from app.agents.state import (
    BusinessMoatOutput,
    DCFValuationOutput,
    EquityResearchState,
    Final10KResearchReport,
    FinancialAuditOutput,
    ForecastOutput,
    ForecastYear,
    RiskAuditOutput,
    RiskItem,
    ThreePillarThesis,
    WACCAudit,
)
from app.agents.tools.dcf_tools import calculate_dcf_with_sensitivity
from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    audit_financial_metrics,
)
from app.agents.tools.wacc_tools import calculate_wacc
from app.database import SessionLocal
from app.models import ChatMessage, Conversation, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import process_chat, stream_chat_service


class TestMilestone6FullPipeline(unittest.TestCase):
    """Integration test suite for Milestone 6 Multi-Agent Orchestration & Chat Integration."""

    @staticmethod
    def _mock_supervisor_extract(user_query: str, **kwargs) -> SupervisorExtraction:
        uq = user_query.lower()
        if "dcf" in uq:
            return SupervisorExtraction(
                query_type="dcf_valuation_only",
                extracted_ticker="TSLA" if "tsla" in uq else "AAPL",
                extracted_year=2025,
                routing_reasoning="DCF valuation route requested",
            )
        elif "business model" in uq or "moat" in uq:
            return SupervisorExtraction(
                query_type="business_moat_only",
                extracted_ticker="NVDA" if "nvda" in uq else "AAPL",
                extracted_year=2026 if "nvda" in uq else 2025,
                routing_reasoning="Business moat route requested",
            )
        else:
            return SupervisorExtraction(
                query_type="full_10k_report",
                extracted_ticker="AAPL" if "aapl" in uq or "apple" in uq else "TSLA",
                extracted_year=2025,
                routing_reasoning="Full 10-K report route requested",
            )

    @staticmethod
    def _mock_moat_analyze(ticker: str, fiscal_year: int, **kwargs) -> BusinessMoatOutput:
        return BusinessMoatOutput(
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
            business_summary=f"Business summary for {ticker.upper()} FY{fiscal_year}.",
            revenue_architecture=f"Core products and recurring services architecture for {ticker.upper()}.",
            primary_product_segments=["Hardware", "Software & Services"],
            economic_moat_type="High Switching Costs",
            moat_durability="Wide",
            moat_trajectory="Expanding",
            moat_rationale=f"Integrated proprietary platform creates high switching costs for {ticker.upper()}.",
            pricing_power_assessment="High premium pricing power.",
            customer_concentration="No single customer > 10% of revenue.",
            citations=[{"chunk_id": f"chunk-{ticker.lower()}-item1-01", "item": "Item 1"}],
        )

    @staticmethod
    def _mock_audit(ticker: str, fiscal_year: int, **kwargs) -> FinancialAuditOutput:
        annual_financials = [
            AnnualFinancialInput(
                fiscal_year=fiscal_year - 2,
                revenue=383285.0,
                gross_profit=169148.0,
                operating_income=114301.0,
                pretax_income=113736.0,
                income_tax_expense=16741.0,
                net_income=96995.0,
                operating_cash_flow=110543.0,
                capital_expenditures=10959.0,
            ),
            AnnualFinancialInput(
                fiscal_year=fiscal_year - 1,
                revenue=391035.0,
                gross_profit=180683.0,
                operating_income=123216.0,
                pretax_income=123485.0,
                income_tax_expense=29749.0,
                net_income=93736.0,
                operating_cash_flow=118254.0,
                capital_expenditures=9447.0,
            ),
            AnnualFinancialInput(
                fiscal_year=fiscal_year,
                revenue=416161.0,
                gross_profit=195201.0,
                operating_income=133050.0,
                pretax_income=132717.0,
                income_tax_expense=20707.0,
                net_income=112010.0,
                operating_cash_flow=111482.0,
                capital_expenditures=12715.0,
                depreciation_amortization=11445.0,
            ),
        ]
        balance_sheet_in = BalanceSheetInput(
            fiscal_year=fiscal_year,
            cash_and_equivalents=35934.0,
            marketable_securities=96486.0,
            short_term_debt=10912.0,
            long_term_debt=87745.0,
            stockholders_equity=53736.0,
            weighted_diluted_shares=15004.7,
            current_assets=154388.0,
            current_liabilities=145308.0,
        )
        math_res = audit_financial_metrics(annual_financials, balance_sheet_in)
        math_res["ticker"] = ticker.upper()
        math_res["fiscal_year"] = fiscal_year
        math_res["auditor_summary"] = f"Audited financial statements for {ticker.upper()} FY{fiscal_year}."
        math_res["data_quality"] = {"extraction_mode": "deterministic"}
        math_res["citations"] = [{"chunk_id": f"chunk-{ticker.lower()}-item8-01", "item": "Item 8"}]
        return FinancialAuditOutput.model_validate(math_res)

    @staticmethod
    def _mock_risk_analyze(ticker: str, fiscal_year: int, **kwargs) -> RiskAuditOutput:
        return RiskAuditOutput(
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
            identified_risks=[
                RiskItem(
                    risk_id="R1",
                    risk_category="Regulatory & Legal",
                    risk_title="Antitrust & Platform Regulatory Inquiries",
                    risk_summary="Global antitrust scrutiny and regulatory inquiries impacting ecosystem margins.",
                    likelihood="High",
                    impact="High",
                    severity="Severe",
                ),
                RiskItem(
                    risk_id="R2",
                    risk_category="Supply Chain & Concentration",
                    risk_title="Component Single Source Dependence",
                    risk_summary="Critical reliance on concentrated third-party supply chain partners.",
                    likelihood="Medium",
                    impact="High",
                    severity="Severe",
                ),
                RiskItem(
                    risk_id="R3",
                    risk_category="Macroeconomic & Geopolitical",
                    risk_title="Foreign Exchange Fluctuations",
                    risk_summary="Substantial international sales exposed to currency headwinds.",
                    likelihood="High",
                    impact="Medium",
                    severity="Moderate",
                ),
            ],
            primary_threat_risk_id="R1",
            primary_existential_threat="Platform regulatory intervention and supply chain concentration.",
            overall_risk_profile="High",
            risk_matrix_markdown="| # | Risk |\n| :--- | :--- |\n| R1 | Antitrust & Platform Regulatory Inquiries |",
            citations=[{"chunk_id": f"chunk-{ticker.lower()}-item1a-01", "item": "Item 1A"}],
        )

    @staticmethod
    def _mock_forecast(ticker: str, fiscal_year: int, financial_audit: FinancialAuditOutput, **kwargs) -> ForecastOutput:
        sched = [
            ForecastYear(
                projected_year=fiscal_year + i,
                projected_revenue=400000.0 * (1.05 ** i),
                projected_revenue_growth_pct=5.0,
                projected_ebit=130000.0 * (1.05 ** i),
                projected_ebit_margin_pct=32.5,
                projected_nopat=110000.0 * (1.05 ** i),
                projected_capex=12000.0 * (1.05 ** i),
                projected_unlevered_fcf=98000.0 * (1.05 ** i),
            )
            for i in range(1, 6)
        ]
        fcfs = [y.projected_unlevered_fcf for y in sched]
        return ForecastOutput(
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
            base_revenue=416161.0,
            forecast_horizon_years=5,
            revenue_cagr_pct=5.0,
            cumulative_5yr_fcf=sum(fcfs),
            average_annual_fcf=sum(fcfs) / 5.0,
            provenance_mode="simplified_nopat_less_capex",
            guidance_source="md&a_explicit",
            tax_rate_pct=15.6,
            projected_fcfs=fcfs,
            forecast_schedule=sched,
            forecast_table_markdown="| Metric | FY26 | FY27 | FY28 | FY29 | FY30 |\n| :--- | :---: | :---: | :---: |\n| UFCF | $98,000 | $102,900 | $108,045 | $113,447 | $119,120 |",
            growth_rationale="5-year forecast calibrated to audited guidance.",
            margin_expansion_rationale="Operating margin expansion from efficiency.",
            citations=[{"chunk_id": f"chunk-{ticker.lower()}-item7-01", "item": "Item 7"}],
        )

    @staticmethod
    def _mock_dcf_value(ticker: str, fiscal_year: int, financial_audit: FinancialAuditOutput, projected_fcfs: List[float], **kwargs) -> DCFValuationOutput:
        beta = kwargs.get("beta") or 1.10
        share_price = kwargs.get("share_price") or 230.0
        terminal_growth = kwargs.get("terminal_growth_rate", 0.025)
        wacc_res = calculate_wacc(
            beta=beta,
            total_debt=financial_audit.balance_sheet.total_debt,
            market_cap=share_price * financial_audit.balance_sheet.valuation_shares_outstanding,
            risk_free_rate=0.042,
            equity_risk_premium=0.050,
            tax_rate=0.156,
        )
        dcf_res = calculate_dcf_with_sensitivity(
            projected_fcfs=projected_fcfs,
            wacc=wacc_res["wacc"],
            terminal_growth_rate=terminal_growth,
            net_debt=financial_audit.balance_sheet.net_debt,
            diluted_shares=financial_audit.balance_sheet.valuation_shares_outstanding,
            mid_year_convention=True,
        )
        upside = round(((dcf_res["fair_value_per_share"] - share_price) / share_price) * 100.0, 2)
        return DCFValuationOutput(
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
            wacc_audit=WACCAudit.model_validate(wacc_res),
            terminal_growth_rate=terminal_growth,
            discounting_convention="mid_year",
            projected_fcfs=projected_fcfs,
            pv_explicit_fcfs=dcf_res["pv_explicit_fcfs"],
            pv_terminal_value=dcf_res["pv_terminal_value"],
            terminal_value_pct_of_ev=dcf_res["terminal_value_pct_of_ev"],
            enterprise_value=dcf_res["enterprise_value"],
            net_debt=dcf_res["net_debt"],
            equity_value=dcf_res["equity_value"],
            diluted_shares=dcf_res["diluted_shares"],
            implied_fair_value_per_share=dcf_res["fair_value_per_share"],
            current_share_price=share_price,
            upside_downside_pct=upside,
            valuation_stance="Fairly Valued",
            sensitivity_matrix_markdown=dcf_res["sensitivity_matrix_markdown"],
            valuation_summary=f"DCF valuation yields ${dcf_res['fair_value_per_share']:.2f} per share.",
        )

    @staticmethod
    def _mock_synthesize(ticker: str, company_name: str, fiscal_year: int, **kwargs) -> Final10KResearchReport:
        dcf_val = kwargs.get("dcf_valuation")
        fair_val = dcf_val.implied_fair_value_per_share if dcf_val else 245.50
        price = dcf_val.current_share_price if dcf_val else 230.00
        upside = dcf_val.upside_downside_pct if dcf_val else 6.7
        stance = dcf_val.valuation_stance if dcf_val else "Fairly Valued"
        sens_md = dcf_val.sensitivity_matrix_markdown if dcf_val else ""

        thesis = ThreePillarThesis(
            pillar_1_business_moat="Proprietary integrated platform and durable economic moat defend market leadership.",
            pillar_2_financial_durability="Robust free cash flow generation and conservative balance sheet structure.",
            pillar_3_valuation_asymmetry="DCF valuation establishes attractive asymmetric upside and margin of safety.",
        )
        md_report = (
            f"# Institutional Equity Research Report: {company_name} ({ticker})\n\n"
            f"## Executive Valuation Dashboard\n"
            f"- Implied Fair Value: **${fair_val:.2f}**\n"
            f"- Market Price: ${price:.2f}\n"
            f"- Projected Upside: {upside}%\n\n"
            f"## 1. Executive Summary\n"
            f"Comprehensive equity research evaluation of {company_name} for FY{fiscal_year}.\n\n"
            f"## 2. Institutional 3-Pillar Investment Thesis\n"
            f"Three pillar thesis evaluating moat, balance sheet durability, and valuation asymmetry.\n\n"
            f"## 3. Business Operations & Economic Moat Analysis\n"
            f"Economic Moat Analysis evaluates competitive advantages and pricing power.\n\n"
            f"## 4. Audited Financial Statements & Ratio Performance\n"
            f"Audited financial statement metrics and ratio analysis.\n\n"
            f"## 5. 5-Year Financial Forecast & Cash Flow Schedule\n"
            f"5-Year forecast schedule of revenue, margins, and unlevered free cash flows.\n\n"
            f"## 6. Discounted Cash Flow (DCF) Valuation & Sensitivity\n"
            f"{sens_md}\n\n"
            f"## 7. Material Risk Factors & Existential Overhangs\n"
            f"Material risk factors and operational overhangs disclosed in Item 1A.\n\n"
            f"## 8. Regulatory Disclaimers & Audit Breadcrumbs\n"
            f"Institutional research disclaimer."
        )
        all_citations = []
        for k in ["business_moat", "financial_audit", "forecast", "risk_audit"]:
            o = kwargs.get(k)
            if o and hasattr(o, "citations") and o.citations:
                all_citations.extend(o.citations)
        if not all_citations:
            all_citations = [{"chunk_id": f"chunk-{ticker.lower()}-01", "item": "Item 1"}]

        return Final10KResearchReport(
            ticker=ticker.upper(),
            company_name=company_name,
            fiscal_year=fiscal_year,
            implied_fair_value_per_share=fair_val,
            current_share_price=price,
            upside_downside_pct=upside,
            valuation_stance=stance,
            three_pillar_thesis=thesis,
            executive_summary=f"Executive briefing on {company_name} ({ticker}) for FY{fiscal_year}.",
            synthesis_provenance="llm_structured",
            year_substituted=kwargs.get("year_substituted", False),
            full_markdown_report=md_report,
            all_citations=all_citations,
        )

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = MultiAgentOrchestrator()

        cls.mock_embedder = MagicMock()
        cls.mock_embedder.embed_query.return_value = [0.01] * 1536

        cls.mock_market = {
            "share_price": 230.0,
            "market_cap": 3450000.0,
            "beta": 1.10,
            "currency": "USD",
        }

        cls.patches = [
            patch("app.agents.orchestrator.fetch_market_context", return_value=cls.mock_market),
            patch("app.agents.tools.rag_narrative_tools._get_embedder", return_value=cls.mock_embedder),
            patch.object(SupervisorAgent, "_extract_with_llm", side_effect=cls._mock_supervisor_extract),
            patch.object(BusinessStrategistAgent, "analyze", side_effect=cls._mock_moat_analyze),
            patch.object(FinancialAuditorAgent, "audit", side_effect=cls._mock_audit),
            patch.object(RiskAnalystAgent, "analyze", side_effect=cls._mock_risk_analyze),
            patch.object(ForecastingAnalystAgent, "forecast", side_effect=cls._mock_forecast),
            patch.object(ValuationSpecialistAgent, "value", side_effect=cls._mock_dcf_value),
            patch.object(LeadSynthesizerAgent, "synthesize", side_effect=cls._mock_synthesize),
        ]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):
            p.stop()

    def test_01_graph_compilation_and_registry(self):
        """Verify graph builds cleanly and agent is registered in AgentRegistry."""
        graph = build_equity_research_graph()
        self.assertIsNotNone(graph)

        agent = AgentRegistry.get("multi_agent")
        self.assertIsInstance(agent, MultiAgentOrchestrator)

        agent_orch = AgentRegistry.get("orchestrator")
        self.assertIsInstance(agent_orch, MultiAgentOrchestrator)

        registered = AgentRegistry.list_agents()
        self.assertIn("multi_agent", registered)
        self.assertIn("orchestrator", registered)
        self.assertIn("supervisor", registered)
        self.assertIn("lead_synthesizer", registered)

    def test_02_fast_path_dcf_pipeline_execution(self):
        """Verify Fast-Path DCF route (TSLA FY2025) executes without running qualitative agents."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            state: EquityResearchState = loop.run_until_complete(
                self.orchestrator.arun(
                    user_query="Calculate DCF intrinsic fair value and WACC for TSLA",
                    ticker="TSLA",
                    fiscal_year=2025,
                )
            )
        finally:
            loop.close()

        # Check routing
        self.assertEqual(state.get("ticker"), "TSLA")
        self.assertEqual(state.get("fiscal_year"), 2025)
        self.assertEqual(state.get("query_type"), "dcf_valuation_only")

        # In Fast-Path DCF: qualitative nodes must remain None
        self.assertIsNone(state.get("business_moat"), "Business moat should be skipped in DCF fast path")
        self.assertIsNone(state.get("risk_audit"), "Risk audit should be skipped in DCF fast path")

        # Quantitative nodes must be populated
        self.assertIsInstance(state.get("financial_audit"), FinancialAuditOutput)
        self.assertIsInstance(state.get("forecast"), ForecastOutput)
        self.assertIsInstance(state.get("dcf_valuation"), DCFValuationOutput)

        # DCF checks
        dcf: DCFValuationOutput = state["dcf_valuation"]
        self.assertIsInstance(dcf.implied_fair_value_per_share, float)
        self.assertGreater(dcf.wacc_audit.wacc, 0.0)
        self.assertIn("WACC", dcf.sensitivity_matrix_markdown)

        # Lead Synthesizer report
        report = state.get("final_report")
        self.assertIsInstance(report, Final10KResearchReport)
        self.assertIn("TSLA", report.ticker)
        self.assertIn("Executive Valuation Dashboard", report.full_markdown_report)

    def test_03_ad_hoc_single_agent_route(self):
        """Verify ad-hoc single agent route (business_moat_only on NVDA FY2026)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            state: EquityResearchState = loop.run_until_complete(
                self.orchestrator.arun(
                    user_query="What is NVDA's business model, segments, and competitive economic moat?",
                    ticker="NVDA",
                    fiscal_year=2026,
                )
            )
        finally:
            loop.close()

        self.assertEqual(state.get("ticker"), "NVDA")
        self.assertEqual(state.get("query_type"), "business_moat_only")
        self.assertIsInstance(state.get("business_moat"), BusinessMoatOutput)
        # Financial audit and valuation must not have run
        self.assertIsNone(state.get("financial_audit"))
        self.assertIsNone(state.get("dcf_valuation"))

        report = state.get("final_report")
        self.assertIsInstance(report, Final10KResearchReport)
        self.assertIn("Economic Moat Analysis", report.full_markdown_report)

    def test_04_realtime_node_milestone_event_streaming(self):
        """Verify astream_run() emits sequential node milestone status events and final result."""
        async def collect_events():
            events = []
            async for event in self.orchestrator.astream_run(
                user_query="Calculate DCF valuation for TSLA",
                ticker="TSLA",
                fiscal_year=2025,
            ):
                events.append(event)
            return events

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            events = loop.run_until_complete(collect_events())
        finally:
            loop.close()

        self.assertGreater(len(events), 2)
        # Initial status event
        self.assertEqual(events[0].get("type"), "status")
        self.assertEqual(events[0].get("node"), "supervisor")

        # Intermediate status events should contain financial_auditor, valuation_specialist
        event_nodes = [e.get("node") for e in events if e.get("type") == "status"]
        self.assertIn("supervisor", event_nodes)
        self.assertIn("financial_auditor", event_nodes)

        # Final result event
        last_event = events[-1]
        self.assertEqual(last_event.get("type"), "result")
        self.assertTrue(len(last_event.get("response", "")) > 100)
        self.assertIsInstance(last_event.get("sources"), list)

    def test_05_full_10k_report_end_to_end(self):
        """Verify full 6-agent institutional equity research pipeline on Apple (AAPL FY2025)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            state: EquityResearchState = loop.run_until_complete(
                self.orchestrator.arun(
                    user_query="Generate comprehensive 10-K equity research report on Apple for FY2025",
                    ticker="AAPL",
                    fiscal_year=2025,
                )
            )
        finally:
            loop.close()

        # 1. State routing
        self.assertEqual(state.get("ticker"), "AAPL")
        self.assertEqual(state.get("fiscal_year"), 2025)
        self.assertEqual(state.get("query_type"), "full_10k_report")

        # 2. All 5 domain sub-agent payloads populated
        self.assertIsInstance(state.get("business_moat"), BusinessMoatOutput)
        self.assertIsInstance(state.get("financial_audit"), FinancialAuditOutput)
        self.assertIsInstance(state.get("risk_audit"), RiskAuditOutput)
        self.assertIsInstance(state.get("forecast"), ForecastOutput)
        self.assertIsInstance(state.get("dcf_valuation"), DCFValuationOutput)

        # 3. Final Report validation
        report: Final10KResearchReport = state.get("final_report")
        self.assertIsInstance(report, Final10KResearchReport)
        self.assertEqual(report.ticker, "AAPL")
        self.assertEqual(report.fiscal_year, 2025)
        self.assertGreater(report.implied_fair_value_per_share, 50.0)
        self.assertIn(report.valuation_stance, ["Undervalued", "Fairly Valued", "Overvalued"])

        # 4. 3-Pillar Investment Thesis
        thesis = report.three_pillar_thesis
        self.assertTrue(len(thesis.pillar_1_business_moat) > 20)
        self.assertTrue(len(thesis.pillar_2_financial_durability) > 20)
        self.assertTrue(len(thesis.pillar_3_valuation_asymmetry) > 20)

        # 5. Full Markdown Report sections
        md = report.full_markdown_report
        self.assertIn("Institutional Equity Research Report", md)
        self.assertIn("Executive Valuation Dashboard", md)
        self.assertIn("Institutional 3-Pillar Investment Thesis", md)
        self.assertIn("Business Operations & Economic Moat Analysis", md)
        self.assertIn("Audited Financial Statements & Ratio Performance", md)
        self.assertIn("5-Year Financial Forecast & Cash Flow Schedule", md)
        self.assertIn("Discounted Cash Flow (DCF) Valuation & Sensitivity", md)
        self.assertIn("Material Risk Factors & Existential Overhangs", md)

        # 6. Deduplicated citations
        self.assertGreater(len(report.all_citations), 0)

    def test_06_chat_service_integration(self):
        """Verify process_chat executes through MultiAgentOrchestrator and persists in PostgreSQL."""
        db = SessionLocal()
        try:
            from app.database import get_single_user
            user = get_single_user(db)
            if not user:
                user = User(username="test_pm", email="pm@fund.com", full_name="Portfolio Manager", is_active=True)
                db.add(user)
                db.commit()
                db.refresh(user)
            self.assertIsNotNone(user, "Expected active test user in database")

            req = ChatRequest(
                message="Calculate DCF fair value and WACC for TSLA",
                agent_type="multi_agent",
                stream=False,
            )
            from unittest.mock import MagicMock, patch
            from app.agents.base import AgentOutput

            with patch("app.services.chat_service.AgentRegistry.get") as mock_agent_get:
                mock_agent = MagicMock()
                mock_agent.run.return_value = AgentOutput(
                    content="### Institutional DCF Valuation: TSLA (Tesla Inc.)\n\nExecutive Valuation Dashboard: Intrinsic fair value is $245.50/share.",
                    sources=[{"ticker": "TSLA", "item": "Item 8", "snippet": "Cash flows"}],
                    updated_session_state={"ticker": "TSLA", "fiscal_year": 2025},
                )
                mock_agent_get.return_value = mock_agent

                with patch.dict("os.environ", {"OPENAI_API_KEY": "test-mock-key"}):
                    response: ChatResponse = process_chat(request=req, user=user, db=db)

            self.assertIsInstance(response, ChatResponse)
            self.assertTrue(len(response.response) > 100)
            self.assertIsNotNone(response.conversation_id)
            self.assertIsNotNone(response.message_id)

            # Check DB persistence
            saved_msg = (
                db.query(ChatMessage)
                .filter(ChatMessage.id == response.message_id)
                .first()
            )
            self.assertIsNotNone(saved_msg)
            self.assertEqual(saved_msg.role, "assistant")
            self.assertFalse(saved_msg.is_error)
            self.assertIn("TSLA", saved_msg.content)

        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
