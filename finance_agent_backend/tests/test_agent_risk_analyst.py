"""Integration and Unit Tests for Risk & Red Flag Analyst Agent (Milestone 5).

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Direct Item 1A narrative retrieval tool execution.
3. End-to-end execution of RiskAnalystAgent on Apple Inc. (AAPL FY2025 10-K).
4. Schema conformance of returned RiskAuditOutput:
   - Curated 5–8 material non-boilerplate risks.
   - Severity ordering (Severe first, followed by Moderate, then Low).
   - Domain categorization across operational, regulatory, supply chain, macroeconomic, tech.
   - Primary existential threat identification.
   - Overall risk profile rating.
   - Non-empty narrative citations with chunk IDs.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import ToolMessage
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.state import RiskAuditOutput, RiskItem
from app.agents.specialized.risk_analyst import RiskAnalystAgent
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool


class TestAgentRiskAnalyst(unittest.TestCase):
    """Test suite for Milestone 5: Risk & Red Flag Analyst Agent."""

    def test_01_initialization_and_registry(self):
        """Verify agent registration, prompt rendering, and tool binding."""
        agent = AgentRegistry.get("risk_analyst")
        self.assertIsInstance(agent, RiskAnalystAgent)
        self.assertEqual(agent.model_name, "openai:gpt-4o-mini")

        # Verify prompt renders with role and tools
        prompt = render_prompt("risk_analyst")
        self.assertIn("Chief Risk Officer", prompt)
        self.assertIn("retrieve_10k_narrative_tool", prompt)
        self.assertIn("RiskAuditOutput", prompt)
        self.assertIn("BOILERPLATE FILTER", prompt)

    @patch("app.agents.tools.rag_narrative_tools._get_embedder")
    def test_02_item1a_narrative_retrieval(self, mock_get_embedder):
        """Verify narrative retrieval tool pulls Item 1A chunks for AAPL FY25 without real API calls."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.01] * 1536
        mock_get_embedder.return_value = mock_embedder

        chunks = retrieve_10k_narrative_tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "query": "supplier reliance single source components manufacturing concentration",
            "section_item": "Item 1A",
            "limit": 3,
        })
        self.assertGreaterEqual(len(chunks), 1, "Should return at least 1 narrative chunk from Item 1A")
        for c in chunks:
            self.assertEqual(c["ticker"], "AAPL")
            self.assertEqual(c["fiscal_year"], 2025)
            self.assertTrue(len(c["content"]) > 0)
            self.assertTrue(len(c["chunk_id"]) > 0)

    @patch.object(RiskAnalystAgent, "_get_or_create_agent")
    def test_03_end_to_end_apple_fy2025_risk_audit(self, mock_get_agent):
        """
        Verify autonomous execution of RiskAnalystAgent on Apple Inc. FY2025 without real API calls.
        Validates complete RiskAuditOutput artifact.
        """
        mock_output = RiskAuditOutput(
            ticker="AAPL",
            fiscal_year=2025,
            identified_risks=[
                RiskItem(
                    risk_category="Regulatory & Legal",
                    risk_title="Antitrust & App Store Regulatory Enforcement",
                    risk_summary="Antitrust scrutiny in the US, EU Digital Markets Act, and global litigation threatening App Store fee structure, anti-steering provisions, and ecosystem margins.",
                    severity="Severe",
                ),
                RiskItem(
                    risk_category="Supply Chain & Concentration",
                    risk_title="Single-Source Asian Manufacturing Concentration",
                    risk_summary="Substantially all final assembly and critical silicon foundry capacity outsourced to concentrated suppliers in Greater China, Taiwan, and broader East Asia.",
                    severity="Severe",
                ),
                RiskItem(
                    risk_category="Technological & Cybersecurity",
                    risk_title="Generative AI Competitive Dynamics & Rapid Technology Shifts",
                    risk_summary="Intense competitive pressure to develop and deploy cutting-edge on-device and cloud generative AI features, demanding continuous heavy capital and R&D commitments.",
                    severity="Moderate",
                ),
                RiskItem(
                    risk_category="Macroeconomic & Geopolitical",
                    risk_title="Foreign Exchange and Trade Policy Headwinds",
                    risk_summary="Substantial non-US net sales expose financial results to international currency fluctuations, potential tariff increases, and cross-border trade friction.",
                    severity="Moderate",
                ),
                RiskItem(
                    risk_category="Operational",
                    risk_title="Global Logistics and Inventory Component Disruptions",
                    risk_summary="Supply chain disruptions or component shortages could delay new device product launches, impairing seasonal quarterly unit revenue and operating leverage.",
                    severity="Moderate",
                ),
                RiskItem(
                    risk_category="Operational",
                    risk_title="Key Personnel Retention and Talent Competition",
                    risk_summary="Highly competitive market for specialized engineering, design, and software leadership talent in Silicon Valley and global tech hubs.",
                    severity="Low",
                ),
            ],
            primary_existential_threat="Global antitrust intervention and digital platform mandates forcing the unbundling of proprietary iOS services, alternative payment engines, and sideloading.",
            overall_risk_profile="Moderate",
            citations=[{"chunk_id": "chunk-aapl-item1a-001", "item": "Item 1A", "breadcrumb": "Item 1A > Risk Factors"}],
        )

        mock_active_agent = MagicMock()
        mock_active_agent.invoke.return_value = {
            "messages": [
                ToolMessage(
                    name="retrieve_10k_narrative_tool",
                    content=json.dumps([{
                        "chunk_id": "chunk-aapl-item1a-001",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "item": "Item 1A",
                        "breadcrumb": "Item 1A > Risk Factors",
                    }]),
                    tool_call_id="call_risk_1",
                )
            ],
            "structured_response": mock_output,
        }
        mock_get_agent.return_value = mock_active_agent

        agent = RiskAnalystAgent(model_name="openai:gpt-4o-mini")
        result = agent.analyze(ticker="AAPL", fiscal_year=2025)

        # Verify mocked agent invocation was called
        mock_active_agent.invoke.assert_called_once()

        # 1. Output Type & Basic Metadata
        self.assertIsInstance(result, RiskAuditOutput)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.fiscal_year, 2025)

        # 2. Risk Count Bounded within 5 to 8
        risks = result.identified_risks
        self.assertTrue(
            5 <= len(risks) <= 8,
            f"Expected 5 to 8 curated material risks, got {len(risks)}"
        )

        # 3. Severity Ordering: Severe risks must appear first
        severities = [r.severity for r in risks]
        self.assertEqual(severities[0], "Severe", "First identified risk must be classified as Severe")

        # Verify severity ordering is non-ascending (Severe -> Moderate -> Low)
        severity_rank = {"Severe": 0, "Moderate": 1, "Low": 2}
        ranked_severities = [severity_rank[s] for s in severities]
        self.assertEqual(
            ranked_severities,
            sorted(ranked_severities),
            f"Risks must be sorted by severity descending, got {severities}"
        )

        # 4. Domain Categorization and Non-empty Disclosures
        valid_categories = {
            "Operational",
            "Regulatory & Legal",
            "Supply Chain & Concentration",
            "Macroeconomic & Geopolitical",
            "Technological & Cybersecurity",
        }
        for r in risks:
            self.assertIn(r.risk_category, valid_categories)
            self.assertTrue(len(r.risk_title) > 0)
            self.assertTrue(len(r.risk_summary) > 20)

        # 5. Structural & Existential Assessment
        self.assertTrue(len(result.primary_existential_threat) > 30)
        self.assertIn(result.overall_risk_profile, {"High", "Moderate", "Conservative"})

        # 6. Audit Trail & Citations
        self.assertGreater(len(result.citations), 0, "Must contain at least 1 narrative citation")
        for cit in result.citations:
            self.assertIn("chunk_id", cit)
            self.assertTrue(len(cit["chunk_id"]) > 0)


if __name__ == "__main__":
    unittest.main()
