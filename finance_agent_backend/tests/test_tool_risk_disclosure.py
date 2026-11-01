"""Unit Tests for the deterministic Risk-Disclosure toolkit (Item 1A).

Validates the aggregate risk profile and the markdown renderers. No database or network access.
"""

import unittest

from app.agents.specialized.risk_analyst.modules.disclosure_changes.calculate import (
    derive_overall_profile,
    render_risk_evidence_table,
    render_risk_matrix_markdown,
)


class TestOverallProfile(unittest.TestCase):
    def test_overall_profile(self):
        self.assertEqual(derive_overall_profile(["Severe", "Severe", "Low"]), "High")
        self.assertEqual(derive_overall_profile(["Severe", "Low", "Low"]), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate"] * 3), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate", "Low"]), "Low")
        self.assertEqual(derive_overall_profile([]), "Not Assessed")


class TestRenderers(unittest.TestCase):
    def test_risk_matrix_and_evidence_table(self):
        risks = [{
            "risk_id": "R1", "risk_title": "Customer | concentration", "risk_category": "Supply Chain & Concentration",
            "likelihood": "High", "impact": "High", "severity": "Severe",
            "financial_transmission": ["Revenue", "Gross Margin"], "disclosure_change": "Unchanged",
        }]
        matrix = render_risk_matrix_markdown(risks)
        self.assertIn("| # | Risk | Category | Likelihood | Impact | Severity | Model line(s) hit | vs prior year |", matrix)
        self.assertIn("| R1 | Customer / concentration | Supply Chain & Concentration | High | High | **Severe** | Revenue, Gross Margin | Unchanged |", matrix)
        self.assertIn("No risks survived", render_risk_matrix_markdown([]))

        table = render_risk_evidence_table([{"chunk_id": "c1", "item": "Item 15", "breadcrumb": "TEST > Item 15"}])
        self.assertIn("| `c1` | Item 15 | TEST > Item 15 |", table)
        self.assertIn("No verified filing evidence", render_risk_evidence_table([]))


if __name__ == "__main__":
    unittest.main()
