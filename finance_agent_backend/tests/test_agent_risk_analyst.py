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

import unittest
from app.agents.registry import AgentRegistry
from app.agents.prompts import render_prompt
from app.agents.state import RiskAuditOutput
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

    def test_02_item1a_narrative_retrieval(self):
        """Verify narrative retrieval tool pulls Item 1A chunks for AAPL FY25."""
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

    def test_03_end_to_end_apple_fy2025_risk_audit(self):
        """
        Verify autonomous execution of RiskAnalystAgent on Apple Inc. FY2025.
        Validates complete RiskAuditOutput artifact.
        """
        agent = RiskAnalystAgent(model_name="openai:gpt-4o-mini")
        result = agent.analyze(ticker="AAPL", fiscal_year=2025)

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
