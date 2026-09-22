"""Unit & Integration Tests for Supervisor & Intent Router Agent.

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Catalog resolution with explicit substitution flagging (year_substituted).
3. Entity extraction without making actual external API calls.
4. Keyword and intent classification.
5. Interactive year confirmation: When requested year is missing, prompts user with latest year.
6. Multi-turn chat confirmation: User agreeing in natural language (e.g. 'sounds great') confirms and executes.
7. Multi-turn chat cancellation: User declining in natural language (e.g. 'nah cancel') cancels cleanly.
8. LLM extraction using mocks (ZERO external API calls).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.supervisor import SupervisorAgent, SupervisorExtraction
from app.agents.state import RoutingPlan


class TestAgentSupervisor(unittest.TestCase):
    """Test suite for Supervisor & Intent Router Agent (100% Mocked - No External API Calls)."""

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

    def test_03_query_entity_extraction_deterministic(self):
        """Verify entity extraction from query text using database catalog without API calls."""
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

    def test_04_error_on_unknown_ticker(self):
        """Verify descriptive ValueError when requested entity is not in database catalog."""
        with self.assertRaises(ValueError) as ctx:
            self.supervisor.resolve_filing_catalog(ticker="UNKNOWN_CORP_XYZ")
        self.assertIn("No ingested 10-K filings found for ticker 'UNKNOWN_CORP_XYZ'", str(ctx.exception))

    def test_05_intent_classification(self):
        """Verify intent classification across all 5 paths without calling LLM."""
        # Moat
        self.assertEqual(
            self.supervisor.classify_intent("What is Apple's economic moat and business model?"),
            "business_moat_only",
        )
        # Full Report
        self.assertEqual(
            self.supervisor.classify_intent("Generate comprehensive 10-K research report on Apple"),
            "full_10k_report",
        )
        # DCF
        self.assertEqual(
            self.supervisor.classify_intent("Calculate DCF fair value and WACC for TSLA"),
            "dcf_valuation_only",
        )
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

    def test_06_year_not_present_prompts_confirmation(self):
        """Verify that when a requested year is not present in the DB, supervisor halts and requests confirmation."""
        # User asks for TSLA 2022, but DB only has 2025
        plan = self.supervisor.route(
            user_query="What is the DCF fair value of TSLA for 2022?",
            ticker="TSLA",
            fiscal_year=2022,
        )
        self.assertIsInstance(plan, RoutingPlan)
        self.assertTrue(plan.needs_confirmation, "Missing year must flag needs_confirmation=True")
        self.assertEqual(plan.suggested_fiscal_year, 2025)
        self.assertIn("not available in our catalog", plan.confirmation_message)
        self.assertIn("FY2025", plan.confirmation_message)
        self.assertEqual(len(plan.active_agents), 0, "No agents should be scheduled until user confirms")
        self.assertIsNotNone(plan.updated_session_state)
        self.assertEqual(plan.updated_session_state.get("pending_action", {}).get("type"), "confirm_year")

    @patch.object(SupervisorAgent, "_check_confirmation_sentiment")
    def test_07_multiturn_chat_confirmation_affirmative(self, mock_check_sentiment):
        """Verify natural language confirmation handled via session_state & LLM sentiment (ZERO API calls)."""
        from app.agents.specialized.supervisor.agent_supervisor import ConfirmationSentiment

        mock_check_sentiment.return_value = ConfirmationSentiment(sentiment="yes")

        session_state = {
            "active_ticker": "TSLA",
            "active_company": "Tesla, Inc.",
            "active_fiscal_year": 2025,
            "last_query_type": "dcf_valuation_only",
            "pending_action": {
                "type": "confirm_year",
                "ticker": "TSLA",
                "suggested_year": 2025,
                "requested_year": 2022,
                "original_query_type": "dcf_valuation_only",
            },
        }

        plan = self.supervisor.route(
            user_query="sounds great, let's do it",
            session_state=session_state,
        )

        self.assertIsInstance(plan, RoutingPlan)
        self.assertEqual(plan.ticker, "TSLA")
        self.assertEqual(plan.fiscal_year, 2025)
        self.assertFalse(plan.needs_confirmation, "Confirmed turn must set needs_confirmation=False")
        self.assertTrue(plan.year_substituted)
        self.assertEqual(plan.query_type, "dcf_valuation_only", "Must preserve original DCF query type from Turn 1")
        self.assertTrue(len(plan.active_agents) > 0, "Agents must now be scheduled to execute")
        self.assertIsNone(plan.updated_session_state.get("pending_action"))
        self.assertTrue(mock_check_sentiment.called)

    @patch.object(SupervisorAgent, "_check_confirmation_sentiment")
    def test_08_multiturn_chat_confirmation_declined(self, mock_check_sentiment):
        """Verify natural language cancellation handled via session_state & LLM sentiment (ZERO API calls)."""
        from app.agents.specialized.supervisor.agent_supervisor import ConfirmationSentiment

        mock_check_sentiment.return_value = ConfirmationSentiment(sentiment="no")

        session_state = {
            "active_ticker": "AAPL",
            "active_company": "Apple Inc.",
            "active_fiscal_year": 2025,
            "last_query_type": "full_10k_report",
            "pending_action": {
                "type": "confirm_year",
                "ticker": "AAPL",
                "suggested_year": 2025,
                "requested_year": 2020,
                "original_query_type": "full_10k_report",
            },
        }

        plan = self.supervisor.route(
            user_query="nah cancel that",
            session_state=session_state,
        )

        self.assertIsInstance(plan, RoutingPlan)
        self.assertTrue(plan.needs_confirmation)
        self.assertIn("cancelled", plan.confirmation_message)
        self.assertEqual(len(plan.active_agents), 0, "No agents should run on cancellation")
        self.assertIsNone(plan.updated_session_state.get("pending_action"))
        self.assertTrue(mock_check_sentiment.called)

    @patch.object(SupervisorAgent, "_get_llm")
    def test_09_mocked_llm_fallback_for_conversational_queries(self, mock_get_llm):
        """Verify conversational entity and intent extraction using a mocked LLM (ZERO API calls)."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorExtraction(
            query_type="business_moat_only",
            extracted_ticker="AAPL",
            extracted_year=2025,
            routing_reasoning="User asking about competitive moat of iPhone maker",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        plan = self.supervisor.route(
            user_query="Can you analyze the economic moat of the iPhone maker?"
        )

        self.assertEqual(plan.ticker, "AAPL")
        self.assertEqual(plan.fiscal_year, 2025)
        self.assertEqual(plan.query_type, "business_moat_only")
        self.assertTrue(mock_structured.invoke.called)

    def test_10_run_interface_returns_confirmation_content_and_session_state(self):
        """Verify BaseAgent.run() interface returns confirmation text and pending_action in session_state."""
        output = self.supervisor.run([
            {"role": "user", "content": "What is the DCF valuation of TSLA for 2019?"}
        ])
        self.assertTrue(len(output.content) > 0)
        self.assertIn("not available in our catalog", output.content)
        self.assertIn("Would you like to proceed with", output.content)
        self.assertEqual(output.sources, [])
        self.assertIsNotNone(output.updated_session_state)
        self.assertEqual(output.updated_session_state.get("pending_action", {}).get("type"), "confirm_year")


if __name__ == "__main__":
    unittest.main()
