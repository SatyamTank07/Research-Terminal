"""Risk output pieces built from the modules' results."""

import unittest

from app.agents.specialized.risk_analyst.assemble import render_risk_evidence_table


class TestEvidenceTable(unittest.TestCase):
    def test_evidence_table(self):
        table = render_risk_evidence_table([{"chunk_id": "c1", "item": "Item 15", "breadcrumb": "TEST > Item 15"}])
        self.assertIn("| `c1` | Item 15 | TEST > Item 15 |", table)
        self.assertIn("No verified filing evidence", render_risk_evidence_table([]))


if __name__ == "__main__":
    unittest.main()
