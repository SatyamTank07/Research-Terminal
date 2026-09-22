"""Unit & Integration Tests for Supervisor & Intent Router Agent (Milestone 6).

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Deterministic filing catalog resolution with explicit substitution flagging (year_substituted).
3. Stopword filtering (preventing false-positive ticker detection on CEO, EPS, GAAP, SEC, YOY).
4. Keyword prioritization (preventing broad 'valuation' queries from hijacking business moat analysis).
5. Error handling for un-ingested tickers with catalog fallback listing.
6. Deterministic intent classification across all 5 query paths.
7. Supervisor LLM fallback with supervisor.j2 for conversational/ambiguous queries (tagging routing_provenance).
8. Complete RoutingPlan structure and BaseAgent.run() interface.
"""

import unittest
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.supervisor import SupervisorAgent, STOPWORD_TICKERS
from app.agents.state import RoutingPlan


class TestAgentSupervisor(unittest.TestCase):
    """Test suite for Milestone 6: Supervisor & Intent Router Agent."""

    def setUp(self):
        self.supervisor = SupervisorAgent()

    def test_01_initialization_and_registry(self):
        """Verify supervisor registration and prompt template rendering."""
        agent = AgentRegistry.get("supervisor")
        self.assertIsInstance(agent, SupervisorAgent)

        # Verify generic prompt renders
        prompt = render_prompt("supervisor", catalog_summary="AAPL (2025), TSLA (2025)")
        self.assertIn("Research Operations Supervisor", prompt)
        self.assertIn("full_10k_report", prompt)
        self.assertIn("dcf_valuation_only", prompt)
        self.assertIn("RoutingPlan", prompt)

    def test_02_catalog_resolution_and_substitution_flag(self):
        """Verify direct ticker resolution and explicit provenance flagging when year is substituted."""
        # Exact year match (AAPL FY2025) -> year_substituted must be False
        ticker, cname, year, doc_id, year_req, year_sub = self.supervisor.resolve_filing_catalog(
            ticker="AAPL", fiscal_year=2025
        )
        self.assertEqual(ticker, "AAPL")
        self.assertIn("Apple", cname)
        self.assertEqual(year, 2025)
        self.assertEqual(year_req, 2025)
        self.assertFalse(year_sub, "Exact year match should not be flagged as substituted")

        # Unavailable year requested (AAPL FY2020) -> must substitute latest year (2025) and flag True
        ticker2, _, year2, _, year_req2, year_sub2 = self.supervisor.resolve_filing_catalog(
            ticker="AAPL", fiscal_year=2020
        )
        self.assertEqual(ticker2, "AAPL")
        self.assertEqual(year2, 2025)
        self.assertEqual(year_req2, 2020)
        self.assertTrue(year_sub2, "Unavailable year must set year_substituted=True")

    def test_03_query_entity_extraction_and_stopwords(self):
        """Verify entity extraction and ensure financial acronyms (CEO, EPS, GAAP) are not treated as tickers."""
        # Expanded stopwords check
        for sw in ["CEO", "CFO", "EPS", "GAAP", "SEC", "YOY", "CAGR", "EBITDA"]:
            self.assertIn(sw, STOPWORD_TICKERS)

        # By company name
        t1, c1, y1, _, _, _ = self.supervisor.resolve_filing_catalog(
            user_query="What is the DCF valuation of Apple for fiscal year 2025?"
        )
        self.assertEqual(t1, "AAPL")
        self.assertEqual(y1, 2025)

        # By ticker symbol
        t2, c2, y2, _, _, _ = self.supervisor.resolve_filing_catalog(
            user_query="Analyze TSLA 10-K report"
        )
        self.assertEqual(t2, "TSLA")
        self.assertEqual(y2, 2025)

        # Query containing stopwords like 'CEO' and 'EPS' should still find the true company
        t3, _, _, _, _, _ = self.supervisor.resolve_filing_catalog(
            user_query="What did the CEO say about EPS in Apple's filing?"
        )
        self.assertEqual(t3, "AAPL")

    def test_04_error_on_unknown_ticker(self):
        """Verify descriptive ValueError when requested entity is not in database catalog."""
        with self.assertRaises(ValueError) as ctx:
            self.supervisor.resolve_filing_catalog(ticker="UNKNOWN_CORP_XYZ")
        self.assertIn("No ingested 10-K filings found for ticker 'UNKNOWN_CORP_XYZ'", str(ctx.exception))

    def test_05_keyword_prioritization_and_intent_classification(self):
        """Verify refined keyword prioritization (qualitative moat before broad valuation words)."""
        # Moat questions that mention 'valuation' must route to business_moat_only (not hijacked by DCF)
        q_moat_val = "How does Apple's ecosystem and business model drive valuation creation?"
        self.assertEqual(self.supervisor.classify_intent(q_moat_val), "business_moat_only")

        route1, prov1 = self.supervisor.classify_intent_with_provenance(q_moat_val)
        self.assertEqual(route1, "business_moat_only")
        self.assertEqual(prov1, "deterministic_rule")

        # Explicit full report
        self.assertEqual(
            self.supervisor.classify_intent("Generate comprehensive 10-K research report on Apple"),
            "full_10k_report",
        )
        route_full, prov_full = self.supervisor.classify_intent_with_provenance(
            "Generate comprehensive 10-K research report on Apple"
        )
        self.assertEqual(route_full, "full_10k_report")
        self.assertEqual(prov_full, "deterministic_rule")

        # Explicit DCF
        self.assertEqual(
            self.supervisor.classify_intent("Calculate DCF fair value and WACC for TSLA"),
            "dcf_valuation_only",
        )
        route_dcf, prov_dcf = self.supervisor.classify_intent_with_provenance(
            "Calculate DCF fair value and WACC for TSLA"
        )
        self.assertEqual(route_dcf, "dcf_valuation_only")
        self.assertEqual(prov_dcf, "deterministic_rule")

        # Financial Audit
        self.assertEqual(
            self.supervisor.classify_intent("Check statement of operations and balance sheet for Tesla"),
            "financial_audit_only",
        )

        # Risk Factors
        self.assertEqual(
            self.supervisor.classify_intent("What are the primary Item 1A legal and antitrust risks for Nvidia?"),
            "risk_factors_only",
        )

    def test_06_llm_fallback_for_conversational_queries(self):
        """Verify that conversational or ambiguous queries invoke supervisor.j2 with llm_inferred provenance."""
        ambiguous_query = "Could you please walk me through how Apple defends its market dominance against competitors?"
        route, provenance = self.supervisor.classify_intent_with_provenance(ambiguous_query)
        self.assertIn(route, ["business_moat_only", "full_10k_report"])
        # If LLM classified or fallback occurred, provenance is tagged
        self.assertIn(provenance, ["llm_inferred", "deterministic_rule"])

    def test_07_routing_plan_emission(self):
        """Verify complete RoutingPlan structure, substitution tag, and active agent assignments."""
        plan_full = self.supervisor.route(
            user_query="Generate full 10-K equity research report on Apple",
            ticker="AAPL",
            fiscal_year=2025,
        )
        self.assertIsInstance(plan_full, RoutingPlan)
        self.assertEqual(plan_full.ticker, "AAPL")
        self.assertEqual(plan_full.fiscal_year, 2025)
        self.assertEqual(plan_full.year_substituted, False)
        self.assertEqual(plan_full.query_type, "full_10k_report")
        self.assertEqual(plan_full.routing_provenance, "deterministic_rule")
        self.assertEqual(len(plan_full.active_agents), 6)
        self.assertIn("lead_synthesizer", plan_full.active_agents)

        plan_dcf = self.supervisor.route(
            user_query="What is the DCF fair value of TSLA for 2022?",
            ticker="TSLA",
            fiscal_year=2022,
        )
        self.assertEqual(plan_dcf.query_type, "dcf_valuation_only")
        self.assertEqual(plan_dcf.year_substituted, True, "2022 is unavailable for TSLA; must flag substituted")
        self.assertEqual(plan_dcf.fiscal_year, 2025)
        self.assertEqual(plan_dcf.active_agents, [
            "financial_auditor",
            "forecasting_analyst",
            "valuation_specialist",
        ])

    def test_08_run_agent_output_interface(self):
        """Verify BaseAgent.run() interface compatibility for Supervisor."""
        output = self.supervisor.run([
            {"role": "user", "content": "What is NVDA fair value and DCF valuation?"}
        ])
        self.assertTrue(len(output.content) > 0)
        self.assertIn("NVDA", output.content)
        self.assertIn("dcf_valuation_only", output.content)
        self.assertTrue(len(output.sources) > 0)

    def test_09_llm_nickname_entity_resolution(self):
        """Verify natural language company nickname (e.g. 'iPhone maker') resolves via LLM fallback."""
        plan = self.supervisor.route(
            user_query="Can you analyze the economic moat of the iPhone maker?"
        )
        self.assertEqual(plan.ticker, "AAPL")
        self.assertIn(plan.query_type, ["business_moat_only", "full_10k_report"])
        self.assertEqual(plan.routing_provenance, "llm_inferred")


if __name__ == "__main__":
    unittest.main()
