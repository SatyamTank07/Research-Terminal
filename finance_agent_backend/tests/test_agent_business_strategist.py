"""Integration and Unit Tests for Business & Moat Strategist Agent (Milestone 5).

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Direct Item 1 narrative retrieval tool execution (mocked embedder - ZERO external API calls).
3. Execution of BusinessStrategistAgent on Apple Inc. (AAPL FY2025 10-K) (mocked LLM - ZERO external API calls).
4. Schema conformance of returned BusinessMoatOutput:
   - Operating model and revenue architecture.
   - Primary product segments and granular segment_details.
   - Economic moat classification, durability, and trajectory.
   - Pricing power assessment and customer concentration.
   - Non-empty narrative citations with chunk IDs.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import ToolMessage

from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.state import BusinessMoatOutput, SegmentDetail
from app.agents.specialized.business_strategist import BusinessStrategistAgent
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool


class TestAgentBusinessStrategist(unittest.TestCase):
    """Test suite for Milestone 5: Business & Moat Strategist Agent (Mocked - No External API Calls)."""

    def test_01_initialization_and_registry(self):
        """Verify agent registration, prompt rendering, and tool binding."""
        agent = AgentRegistry.get("business_strategist")
        self.assertIsInstance(agent, BusinessStrategistAgent)
        self.assertEqual(agent.model_name, "openai:gpt-4o-mini")

        # Verify prompt renders with role and tools
        prompt = render_prompt("business_strategist")
        self.assertIn("Senior Equity Research Analyst", prompt)
        self.assertIn("retrieve_10k_narrative_tool", prompt)
        self.assertIn("Economic Moat Deconstruction", prompt)
        self.assertIn("BusinessMoatOutput", prompt)

    @patch("app.agents.tools.rag_narrative_tools._get_embedder")
    def test_02_item1_narrative_retrieval(self, mock_get_embedder):
        """Verify narrative retrieval tool pulls Item 1 chunks for AAPL FY25 without real API calls."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.01] * 1536
        mock_get_embedder.return_value = mock_embedder

        chunks = retrieve_10k_narrative_tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "query": "primary products services revenue segments",
            "section_item": "Item 1",
            "limit": 3,
        })
        self.assertGreaterEqual(len(chunks), 1, "Should return at least 1 narrative chunk from Item 1")
        for c in chunks:
            self.assertEqual(c["ticker"], "AAPL")
            self.assertEqual(c["fiscal_year"], 2025)
            self.assertTrue(len(c["content"]) > 0)
            self.assertTrue(len(c["chunk_id"]) > 0)
        mock_embedder.embed_query.assert_called_once_with("primary products services revenue segments")

    @patch.object(BusinessStrategistAgent, "_get_or_create_agent")
    def test_03_end_to_end_apple_fy2025_moat_analysis(self, mock_get_agent):
        """
        Verify execution of BusinessStrategistAgent on Apple Inc. FY2025 using mocked agent.
        Validates complete BusinessMoatOutput artifact with ZERO external API calls.
        """
        mock_output = BusinessMoatOutput(
            ticker="AAPL",
            fiscal_year=2025,
            business_summary="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories, and sells a variety of related services.",
            revenue_architecture="Hardware product sales (iPhone, Mac, iPad, Wearables) complemented by high-margin recurring Services ecosystem revenue.",
            primary_product_segments=["iPhone", "Services", "Mac", "iPad", "Wearables, Home and Accessories"],
            segment_details=[
                SegmentDetail(
                    name="iPhone",
                    description="Flagship smartphone product line integrating hardware, software, and Apple Intelligence.",
                    growth_drivers=["Apple Intelligence", "Replacement cycle acceleration", "5G expansion"],
                ),
                SegmentDetail(
                    name="Services",
                    description="High-margin recurring revenue stream including App Store, iCloud, Apple Pay, and Apple Music.",
                    growth_drivers=["Installed base expansion", "Monetization per active device", "Subscription growth"],
                ),
            ],
            economic_moat_type="High Switching Costs",
            moat_durability="Wide",
            moat_trajectory="Expanding",
            moat_rationale="Proprietary iOS ecosystem creates deep consumer lock-in across 2+ billion active devices, supported by high switching costs, brand intangibles, and network effects.",
            pricing_power_assessment="High premium pricing power evidenced by stable premium hardware ASPs and sustained >45% gross margins.",
            customer_concentration="No single customer accounted for more than 10% of total revenue in FY2025.",
            citations=[{"chunk_id": "chunk-aapl-item1-001", "item": "Item 1", "breadcrumb": "Item 1 > Business"}],
        )

        mock_active_agent = MagicMock()
        mock_active_agent.invoke.return_value = {
            "messages": [
                ToolMessage(
                    name="retrieve_10k_narrative_tool",
                    content=json.dumps([{
                        "chunk_id": "chunk-aapl-item1-001",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "item": "Item 1",
                        "breadcrumb": "Item 1 > Business",
                    }]),
                    tool_call_id="call_1",
                )
            ],
            "structured_response": mock_output,
        }
        mock_get_agent.return_value = mock_active_agent

        agent = BusinessStrategistAgent(model_name="openai:gpt-4o-mini")
        result = agent.analyze(ticker="AAPL", fiscal_year=2025)

        # Verify mocked agent invocation was called
        mock_active_agent.invoke.assert_called_once()

        # 1. Output Type & Basic Metadata
        self.assertIsInstance(result, BusinessMoatOutput)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.fiscal_year, 2025)

        # 2. Business Architecture & Segments
        self.assertTrue(len(result.business_summary) > 50)
        self.assertTrue(len(result.revenue_architecture) > 30)

        # Verify primary product segments include major Apple lines
        segments_lower = [s.lower() for s in result.primary_product_segments]
        self.assertGreaterEqual(len(segments_lower), 3, "Apple should have at least 3 primary product segments")
        has_phone = any("iphone" in s or "smartphone" in s or "phone" in s for s in segments_lower)
        has_services = any("service" in s for s in segments_lower)
        self.assertTrue(has_phone, f"Expected phone/smartphone segment, got: {result.primary_product_segments}")
        self.assertTrue(has_services, f"Expected 'Services' in segments, got: {result.primary_product_segments}")

        # Verify segment details are populated
        self.assertGreaterEqual(len(result.segment_details), 2)
        for detail in result.segment_details:
            self.assertTrue(len(detail.name) > 0)
            self.assertTrue(len(detail.description) > 0)

        # 3. Economic Moat Evaluation
        valid_moat_types = {
            "Network Effects",
            "Cost Advantage",
            "High Switching Costs",
            "Intangible Assets / Brand",
            "Efficient Scale",
            "None",
        }
        self.assertIn(result.economic_moat_type, valid_moat_types)
        self.assertIn(result.moat_durability, {"Wide", "Narrow", "None"})
        self.assertIn(result.moat_trajectory, {"Expanding", "Stable", "Deteriorating"})
        self.assertTrue(len(result.moat_rationale) > 50)

        # 4. Market Power Disclosures
        self.assertTrue(len(result.pricing_power_assessment) > 20)
        self.assertTrue(len(result.customer_concentration) > 10)

        # 5. Audit Trail & Citations
        self.assertGreater(len(result.citations), 0, "Must contain at least 1 narrative citation")
        for cit in result.citations:
            self.assertIn("chunk_id", cit)
            self.assertTrue(len(cit["chunk_id"]) > 0)


if __name__ == "__main__":
    unittest.main()

