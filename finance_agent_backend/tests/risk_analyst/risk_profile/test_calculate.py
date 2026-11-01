"""Risk profile: rank order and the 5-8 band, the primary threat among the most severe, the overall profile."""

import unittest

from app.agents.specialized.risk_analyst.modules.risk_inventory import RiskItem
from app.agents.specialized.risk_analyst.modules.risk_profile import (
    derive_overall_profile,
    rank_risks,
    resolve_primary_threat,
)


def _risk(title: str, severity: str, impact: str = "Medium", likelihood: str = "Medium") -> RiskItem:
    return RiskItem(risk_category="Operational", risk_title=title, risk_summary=f"{title} summary.",
                    likelihood=likelihood, impact=impact, severity=severity)


class TestRankRisks(unittest.TestCase):
    def test_most_severe_first_and_renumbered(self):
        low, severe, moderate = _risk("low", "Low", "Low"), _risk("severe", "Severe", "High"), _risk("mod", "Moderate")
        ranked = rank_risks([low, severe, moderate, _risk("m2", "Moderate"), _risk("m3", "Moderate")])
        self.assertEqual([r.risk_title for r in ranked.risks][:3], ["severe", "mod", "m2"])
        self.assertEqual([r.risk_id for r in ranked.risks], ["R1", "R2", "R3", "R4", "R5"])
        self.assertEqual(ranked.notes, ())

    def test_more_than_eight_keeps_the_top_eight(self):
        risks = [_risk(f"r{i}", "Moderate") for i in range(9)] + [_risk("lowest", "Low", "Low", "Low")]
        ranked = rank_risks(risks)
        self.assertEqual(len(ranked.risks), 8)
        self.assertNotIn("lowest", [r.risk_title for r in ranked.risks])
        self.assertEqual(ranked.notes, ("2 lower-ranked verified risks omitted (cap of 8).",))

    def test_fewer_than_five_is_noted(self):
        self.assertEqual(rank_risks([_risk("a", "Low")]).notes,
                         ("Only 1 risks survived evidence verification (target 5-8).",))


class TestPrimaryThreat(unittest.TestCase):
    def setUp(self):
        self.top, self.next_top, self.lower = _risk("top", "Severe"), _risk("next", "Severe"), _risk("low", "Low")
        self.ranked = rank_risks([self.top, self.next_top, self.lower]).risks

    def test_a_choice_among_the_most_severe_is_kept_with_its_statement(self):
        threat = resolve_primary_threat(self.ranked, self.next_top, {}, "R7", "Why it matters.")
        self.assertEqual((threat.risk_id, threat.statement, threat.note), ("R2", "Why it matters.", None))

    def test_a_kept_choice_without_a_statement_gets_title_and_summary(self):
        self.assertEqual(resolve_primary_threat(self.ranked, self.top, {}, "R1", "").statement, "top: top summary.")

    def test_a_merged_choice_follows_the_risk_that_absorbed_it(self):
        merged = _risk("merged away", "Low")
        threat = resolve_primary_threat(self.ranked, merged, {id(merged): self.next_top}, "R9", "")
        self.assertEqual((threat.risk_id, threat.note), ("R2", None))

    def test_a_less_severe_choice_is_re_pointed_to_the_top_risk(self):
        threat = resolve_primary_threat(self.ranked, self.lower, {}, "R3", "Lower risk.")
        self.assertEqual((threat.risk_id, threat.statement), ("R1", "top: top summary."))
        self.assertEqual(threat.note, "Primary threat re-pointed to R1 ('top'): the model's choice ('low') "
                                      "was not among the most severe verified risks.")

    def test_a_choice_that_failed_verification_is_re_pointed(self):
        threat = resolve_primary_threat(self.ranked, None, {}, "R99", "")
        self.assertEqual(threat.note, "Primary threat re-pointed to R1: the model's choice did not survive verification.")
        self.assertIsNone(resolve_primary_threat(self.ranked, None, {}, None, "").note)

    def test_no_risks_means_no_primary_threat(self):
        self.assertEqual(resolve_primary_threat([], self.top, {}, "R1", "x"), resolve_primary_threat([], None, {}, None, ""))
        self.assertIsNone(resolve_primary_threat([], None, {}, None, "").risk_id)


class TestOverallProfile(unittest.TestCase):
    def test_overall_profile(self):
        self.assertEqual(derive_overall_profile(["Severe", "Severe", "Low"]), "High")
        self.assertEqual(derive_overall_profile(["Severe", "Low", "Low"]), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate"] * 3), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate", "Low"]), "Low")
        self.assertEqual(derive_overall_profile([]), "Not Assessed")


if __name__ == "__main__":
    unittest.main()
