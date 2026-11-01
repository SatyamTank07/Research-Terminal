"""The verified risk matrix as markdown."""

import unittest

from app.agents.specialized.risk_analyst.modules.risk_profile import render_risk_matrix_markdown, table_cell


class TestRiskMatrix(unittest.TestCase):
    def test_risk_matrix(self):
        risks = [{
            "risk_id": "R1", "risk_title": "Customer | concentration", "risk_category": "Supply Chain & Concentration",
            "likelihood": "High", "impact": "High", "severity": "Severe",
            "financial_transmission": ["Revenue", "Gross Margin"], "disclosure_change": "Unchanged",
        }]
        matrix = render_risk_matrix_markdown(risks)
        self.assertIn("| # | Risk | Category | Likelihood | Impact | Severity | Model line(s) hit | vs prior year |", matrix)
        self.assertIn("| R1 | Customer / concentration | Supply Chain & Concentration | High | High | **Severe** | Revenue, Gross Margin | Unchanged |", matrix)
        self.assertIn("No risks survived", render_risk_matrix_markdown([]))

    def test_empty_cells_show_a_dash_and_lose_line_breaks(self):
        self.assertEqual(table_cell(None), "—")
        self.assertEqual(table_cell("a\nb"), "a b")


if __name__ == "__main__":
    unittest.main()
