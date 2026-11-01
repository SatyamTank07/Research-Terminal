"""Risk output assembled from the modules' results: counts, note order, citations, tables, the not-provided output."""

import unittest

from app.agents.specialized.risk_analyst import RiskAuditOutput, RiskDataQuality, RiskEvidence, RiskItem
from app.agents.specialized.risk_analyst.assemble import (
    NOT_PROVIDED_MATRIX,
    assemble_not_provided,
    assemble_risk_audit,
    render_risk_evidence_table,
)
from app.agents.specialized.risk_analyst.modules.quantified_exposures import QuantifiedCoverage, QuantifiedDisclosure
from app.agents.specialized.risk_analyst.modules.risk_inventory import InventoryResult
from app.agents.specialized.risk_analyst.modules.risk_profile import PrimaryThreat, RankedRisks
from tests.risk_analyst.item1a_fixtures import item1a_chunk

POOL = {
    "c1": {"chunk_id": "c1", "ticker": "TEST", "fiscal_year": 2026, "item": "Item 1A", "breadcrumb": "TEST > Item 1A"},
    "c2": {"chunk_id": "c2", "ticker": "TEST", "fiscal_year": 2026, "item": "Item 3", "breadcrumb": "TEST > Item 3"},
}


def _risk(risk_id: str, chunk_ids, severity: str = "Severe", mitigation: str = None) -> RiskItem:
    return RiskItem(
        risk_id=risk_id, risk_category="Operational", risk_title=f"Risk {risk_id}", risk_summary="Exposure.",
        likelihood="High", impact="High", severity=severity,
        evidence=[RiskEvidence(chunk_id=cid, quote="a verified quote") for cid in chunk_ids],
        mitigation_evidence=RiskEvidence(chunk_id=mitigation, quote="a mitigation") if mitigation else None,
    )


class TestAssembleRiskAudit(unittest.TestCase):
    def test_counts_notes_and_fields_in_report_order(self):
        risks = [_risk("R1", ["c1"], mitigation="c2"), _risk("R2", ["c1", "c2"])]
        dq = RiskDataQuality(risks_proposed=4, quotes_rejected=0, notes=["setup note", "llm note"])
        out = assemble_risk_audit(
            RiskAuditOutput(ticker="TEST", fiscal_year=2026),
            InventoryResult(survivors=risks, absorbed={1: risks[0]}, dropped_unverified=1, quotes_rejected=3),
            RankedRisks(risks=risks, notes=("Only 2 risks survived evidence verification (target 5-8).",)),
            QuantifiedCoverage(uncovered=(QuantifiedDisclosure(chunk_id="c1", text="22% of revenue"),),
                               listed=("22% of revenue",), note="1 of 3 quantified Item 1A disclosures are not cited."),
            PrimaryThreat(risk_id="R1", statement="Risk R1: Exposure.", note="Primary threat re-pointed to R1."),
            POOL, None, dq,
        )
        self.assertEqual(out.data_quality.notes, [
            "setup note", "llm note", "Only 2 risks survived evidence verification (target 5-8).",
            "1 of 3 quantified Item 1A disclosures are not cited.", "Primary threat re-pointed to R1.",
        ])
        self.assertEqual((out.data_quality.quotes_rejected, out.data_quality.risks_dropped_unverified,
                          out.data_quality.risks_merged), (3, 1, 1))
        self.assertEqual(out.data_quality.quantified_uncovered, ["22% of revenue"])
        self.assertEqual((out.primary_threat_risk_id, out.primary_existential_threat), ("R1", "Risk R1: Exposure."))
        self.assertEqual(out.overall_risk_profile, "High")
        self.assertEqual([c["chunk_id"] for c in out.citations], ["c1", "c2"])
        self.assertEqual(out.citations[1], POOL["c2"])
        self.assertIn("| R1 | Risk R1 |", out.risk_matrix_markdown)
        self.assertIn("| `c2` | Item 3 | TEST > Item 3 |", out.evidence_table_markdown)


class TestNotProvided(unittest.TestCase):
    def test_filing_without_risk_factors_rates_nothing(self):
        out = assemble_not_provided("TEST", 2026, [item1a_chunk("n1", "Not applicable.")], "item_label")
        self.assertEqual(out.identified_risks, [])
        self.assertEqual(out.data_quality.coverage_mode, "not_provided")
        self.assertEqual(out.data_quality.notes,
                         ['The FY2026 10-K provides no Item 1A risk factors (filing states: "Not applicable."); no risks were rated.'])
        self.assertEqual(out.risk_matrix_markdown, NOT_PROVIDED_MATRIX)
        self.assertEqual(list(out.citations[0]), ["chunk_id", "ticker", "fiscal_year", "item", "breadcrumb", "sub_section"])


class TestEvidenceTable(unittest.TestCase):
    def test_evidence_table(self):
        table = render_risk_evidence_table([{"chunk_id": "c1", "item": "Item 15", "breadcrumb": "TEST > Item 15"}])
        self.assertIn("| `c1` | Item 15 | TEST > Item 15 |", table)
        self.assertIn("No verified filing evidence", render_risk_evidence_table([]))


if __name__ == "__main__":
    unittest.main()
