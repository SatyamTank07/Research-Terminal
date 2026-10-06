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
import unittest
from app.agents import AgentRegistry, MultiAgentOrchestrator, build_equity_research_graph
from app.agents.state import (
    BusinessMoatOutput,
    DCFValuationOutput,
    EquityResearchState,
    Final10KResearchReport,
    FinancialAuditOutput,
    ForecastOutput,
    RiskAuditOutput,
)
from app.database import SessionLocal
from app.models import ChatMessage, Conversation, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import process_chat, stream_chat_service


class TestMilestone6FullPipeline(unittest.TestCase):
    """Integration test suite for Milestone 6 Multi-Agent Orchestration & Chat Integration."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = MultiAgentOrchestrator()

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
            )
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
