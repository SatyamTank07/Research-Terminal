"""Unit & Integration Tests for Supervisor & Intent Router Agent.

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Catalog resolution with explicit substitution flagging (year_substituted).
3. Entity extraction without making actual external API calls.
4. Intent classification (100% Mocked - No external API calls).
5. Automatic graceful year substitution without halting (Zero-Halting, Issue 2.5).
6. Single-turn conversational response generation (Single LLM Hop, Issue 2).
7. BaseAgent.run() interface directly returning conversational response.
8. LLM extraction using mocks (ZERO external API calls).
9. BaseAgent.run() interface returning JSON routing plan for company analysis.
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
        self.assertIn("Senior Equity Research Director", prompt)
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

    @patch.object(SupervisorAgent, "_extract_with_llm")
    def test_05_intent_classification(self, mock_extract):
        """Verify intent classification across routes using mocked LLM (ZERO API calls)."""
        # Moat
        mock_extract.return_value = SupervisorExtraction(query_type="business_moat_only")
        self.assertEqual(
            self.supervisor.classify_intent("What is Apple's economic moat and business model?"),
            "business_moat_only",
        )
        # Full Report
        mock_extract.return_value = SupervisorExtraction(query_type="full_10k_report")
        self.assertEqual(
            self.supervisor.classify_intent("Generate comprehensive 10-K research report on Apple"),
            "full_10k_report",
        )
        # DCF
        mock_extract.return_value = SupervisorExtraction(query_type="dcf_valuation_only")
        self.assertEqual(
            self.supervisor.classify_intent("Calculate DCF fair value and WACC for TSLA"),
            "dcf_valuation_only",
        )
        # Financial Audit
        mock_extract.return_value = SupervisorExtraction(query_type="financial_audit_only")
        self.assertEqual(
            self.supervisor.classify_intent("Check statement of operations and balance sheet for Tesla"),
            "financial_audit_only",
        )
        # Risk Factors
        mock_extract.return_value = SupervisorExtraction(query_type="risk_factors_only")
        self.assertEqual(
            self.supervisor.classify_intent("What are the primary Item 1A legal and antitrust risks for Nvidia?"),
            "risk_factors_only",
        )

    @patch.object(SupervisorAgent, "_extract_with_llm")
    def test_06_year_not_present_auto_substitutes_latest_without_halting(self, mock_extract):
        """Verify that when requested year is missing, supervisor automatically substitutes latest year without halting."""
        mock_extract.return_value = SupervisorExtraction(
            query_type="dcf_valuation_only",
            extracted_ticker="TSLA",
            extracted_year=2022,
        )
        # User asks for TSLA 2022, catalog only has 2025
        plan = self.supervisor.route(
            user_query="What is the DCF fair value of TSLA for 2022?",
            ticker="TSLA",
            fiscal_year=2022,
        )
        self.assertIsInstance(plan, RoutingPlan)
        self.assertFalse(plan.needs_confirmation, "Must not halt for confirmation")
        self.assertTrue(plan.year_substituted, "Must flag year_substituted=True")
        self.assertEqual(plan.fiscal_year, 2025)
        self.assertEqual(plan.year_requested, 2022)
        self.assertGreater(len(plan.active_agents), 0, "Agents must be immediately scheduled")
        self.assertIn("financial_auditor", plan.active_agents)
        self.assertIsNone(plan.updated_session_state.get("pending_action"))

    @patch.object(SupervisorAgent, "_extract_with_llm")
    def test_07_conversational_single_hop_response_generation(self, mock_extract):
        """Verify conversational query generates response in single LLM turn and schedules no sub-agents."""
        mock_extract.return_value = SupervisorExtraction(
            query_type="conversational",
            conversational_response="Weighted Average Cost of Capital (WACC) represents a firm's blended hurdle rate.",
            routing_reasoning="User asking general financial education question",
        )

        plan = self.supervisor.route(
            user_query="What is WACC and how is it used?",
        )

        self.assertIsInstance(plan, RoutingPlan)
        self.assertEqual(plan.query_type, "conversational")
        self.assertEqual(len(plan.active_agents), 0, "No specialized sub-agents needed for conversational")
        self.assertIsNotNone(plan.conversational_response)
        self.assertIn("Weighted Average Cost of Capital", plan.conversational_response)
        self.assertFalse(plan.needs_confirmation)

    @patch.object(SupervisorAgent, "_extract_with_llm")
    def test_08_run_interface_conversational_output(self, mock_extract):
        """Verify BaseAgent.run() interface directly returns conversational_response text."""
        mock_extract.return_value = SupervisorExtraction(
            query_type="conversational",
            conversational_response="Hello! I am your Senior Equity Research Director. I can analyze SEC 10-K filings.",
            routing_reasoning="User greeting",
        )

        output = self.supervisor.run([
            {"role": "user", "content": "Hello, what can you do?"}
        ])
        self.assertEqual(output.content, "Hello! I am your Senior Equity Research Director. I can analyze SEC 10-K filings.")
        self.assertEqual(output.sources, [])

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

    @patch.object(SupervisorAgent, "_extract_with_llm")
    def test_10_run_interface_returns_routing_plan_json(self, mock_extract):
        """Verify BaseAgent.run() interface returns valid JSON routing plan for analysis queries."""
        mock_extract.return_value = SupervisorExtraction(
            query_type="dcf_valuation_only",
            extracted_ticker="TSLA",
            extracted_year=2025,
            routing_reasoning="User asking for DCF valuation of Tesla",
        )
        output = self.supervisor.run([
            {"role": "user", "content": "What is the DCF valuation of TSLA for 2025?"}
        ])
        self.assertTrue(len(output.content) > 0)
        parsed = json.loads(output.content)
        self.assertEqual(parsed.get("ticker"), "TSLA")
        self.assertEqual(parsed.get("fiscal_year"), 2025)
        self.assertEqual(parsed.get("query_type"), "dcf_valuation_only")
        self.assertEqual(len(output.sources), 1)

    @patch.object(SupervisorAgent, "_get_llm")
    def test_11_multi_turn_history_payload_construction(self, mock_get_llm):
        """Verify _extract_with_llm constructs messages_payload with HumanMessage and AIMessage history."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorExtraction(
            query_type="dcf_valuation_only",
            extracted_ticker="AAPL",
            extracted_year=2025,
            routing_reasoning="Multi-turn entity resolution",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        messages = [
            {"role": "user", "content": "Run a DCF with an 8% discount rate."},
            {"role": "assistant", "content": "Which company would you like me to value with an 8% discount rate?"},
            {"role": "user", "content": "Apple"},
        ]

        self.supervisor._extract_with_llm(
            user_query="Apple",
            messages=messages,
        )

        self.assertTrue(mock_structured.invoke.called)
        invoked_payload = mock_structured.invoke.call_args[0][0]

        # Payload must contain: SystemMessage, HumanMessage(Turn 1), AIMessage(Turn 1 response), HumanMessage(Turn 2)
        self.assertEqual(len(invoked_payload), 4)
        self.assertEqual(invoked_payload[1].content, "Run a DCF with an 8% discount rate.")
        self.assertEqual(invoked_payload[2].content, "Which company would you like me to value with an 8% discount rate?")
        self.assertIn("Analyze and route this research inquiry: 'Apple'", invoked_payload[3].content)

    @patch.object(SupervisorAgent, "_get_llm")
    def test_12_long_assistant_report_truncation(self, mock_get_llm):
        """Verify prior long assistant reports (>1500 chars) are truncated with notice."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorExtraction(
            query_type="conversational",
            conversational_response="The operating margin was 30%.",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        long_report = "# Comprehensive 10-K Report\n" + ("Paragraph analysis text. " * 200)
        self.assertGreater(len(long_report), 1500)

        messages = [
            {"role": "user", "content": "Analyze Apple 10-K"},
            {"role": "assistant", "content": long_report},
            {"role": "user", "content": "What was its operating margin?"},
        ]

        self.supervisor._extract_with_llm(
            user_query="What was its operating margin?",
            messages=messages,
        )

        invoked_payload = mock_structured.invoke.call_args[0][0]
        assistant_turn_content = invoked_payload[2].content
        self.assertIn("...[Prior research report excerpted]...", assistant_turn_content)
        self.assertLess(len(assistant_turn_content), len(long_report))

    @patch.object(SupervisorAgent, "_extract_with_llm")
    def test_13_two_turn_clarification_entity_resolution(self, mock_extract):
        """Verify 2-turn clarification flow resolves entity across conversation turns."""
        # Turn 1: User asks for DCF with 8% discount rate, missing company
        mock_extract.return_value = SupervisorExtraction(
            query_type="conversational",
            conversational_response="Which company would you like me to value with an 8% discount rate?",
            routing_reasoning="Missing entity for valuation query",
        )

        turn1_plan = self.supervisor.route(
            user_query="Run DCF with an 8% discount rate",
        )
        self.assertEqual(turn1_plan.query_type, "conversational")

        # Turn 2: User responds "Apple", passing Turn 1 session state and history
        mock_extract.return_value = SupervisorExtraction(
            query_type="dcf_valuation_only",
            extracted_ticker="AAPL",
            extracted_year=2025,
            routing_reasoning="User provided target entity for pending DCF valuation",
        )

        turn2_messages = [
            {"role": "user", "content": "Run DCF with an 8% discount rate"},
            {"role": "assistant", "content": turn1_plan.conversational_response},
            {"role": "user", "content": "Apple"},
        ]

        turn2_plan = self.supervisor.route(
            user_query="Apple",
            messages=turn2_messages,
            session_state=turn1_plan.updated_session_state,
        )

        self.assertEqual(turn2_plan.query_type, "dcf_valuation_only")
        self.assertEqual(turn2_plan.ticker, "AAPL")
        self.assertEqual(turn2_plan.fiscal_year, 2025)

    def test_14_o1_prompt_constant_size(self):
        """Verify supervisor prompt has constant O(1) size and includes anti-interrogation guardrails."""
        prompt = render_prompt("supervisor")
        self.assertIn("MANDATORY CLARIFICATION RULES", prompt)
        self.assertIn("MANDATORY ANTI-INTERROGATION RULES", prompt)
        self.assertIn("Missing Target Entity", prompt)
        self.assertIn("Missing Fiscal Year", prompt)
        self.assertNotIn("Available Ingested 10-K Filings:", prompt)


if __name__ == "__main__":
    unittest.main()
