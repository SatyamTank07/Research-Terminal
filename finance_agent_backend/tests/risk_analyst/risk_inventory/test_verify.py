"""Risk inventory verify: raw proposals sanitized, quotes matched to the filing, exposure / mitigation / impact
screens, and duplicates folded into the higher-ranked risk."""

import unittest

from app.agents.specialized.risk_analyst.modules.risk_inventory import (
    RiskEvidence,
    RiskItem,
    coerce_category,
    is_boilerplate,
    merge_duplicates,
    sanitize_risk,
    states_immaterial,
    substantiates_high_impact,
    verify_evidence,
    verify_inventory,
)
from tests.risk_analyst.item1a_fixtures import CONCENTRATION

POOL = {
    "c1": {"chunk_id": "c1", "item": "Item 1A", "breadcrumb": "TEST > Item 1A", "content": CONCENTRATION},
    "c2": {"chunk_id": "c2", "item": "Item 3", "breadcrumb": "TEST > Item 3",
           "content": "We are party to a patent lawsuit in Texas seeking damages from the company."},
}
CUSTOMER_QUOTE = "sales to one direct customer represented 22% of total revenue"


def _risk(title: str, quotes, likelihood: str = "High", impact: str = "High", **fields) -> RiskItem:
    return RiskItem(
        risk_category=fields.pop("risk_category", "Supply Chain & Concentration"), risk_title=title,
        risk_summary=fields.pop("risk_summary", "A company-specific exposure."), likelihood=likelihood, impact=impact,
        severity="Severe", evidence=[RiskEvidence(chunk_id=cid, quote=q) for cid, q in quotes], **fields,
    )


class TestScreens(unittest.TestCase):
    def test_immateriality_language(self):
        self.assertTrue(states_immaterial(
            "We believe the ultimate outcome will not have a material adverse effect on our operating results."
        ))
        self.assertTrue(states_immaterial("The matter is not expected to have a material impact on our results."))
        self.assertTrue(states_immaterial("We do not believe these claims will be material to our financial position."))
        self.assertFalse(states_immaterial(CONCENTRATION))

    def test_boilerplate_language(self):
        self.assertTrue(is_boilerplate("Economic Conditions and Market Volatility"))
        self.assertTrue(is_boilerplate("Adverse Economic Conditions could reduce demand"))
        self.assertTrue(is_boilerplate("Economic downturns may harm results"))
        self.assertTrue(is_boilerplate("Our stock price may be volatile and could decline"))
        self.assertFalse(is_boilerplate("Dependence on two direct customers for 36% of revenue"))

    def test_high_impact_substantiation(self):
        self.assertTrue(substantiates_high_impact(["anything"], "22% of revenue"))
        self.assertTrue(substantiates_high_impact(["sales to one customer represented 22% of total revenue"], None))
        self.assertTrue(substantiates_high_impact(["We incurred significant costs to remediate the defect"], None))
        self.assertTrue(substantiates_high_impact(["Export restrictions have adversely affected our China revenue"], None))
        self.assertFalse(substantiates_high_impact(["A cyberattack could disrupt our operations and may harm results"], None))
        self.assertFalse(substantiates_high_impact([], None))



class TestSanitize(unittest.TestCase):
    def test_a_proposal_without_title_or_summary_is_discarded(self):
        self.assertIsNone(sanitize_risk({"risk_title": "", "risk_summary": "x"}, 0))
        self.assertIsNone(sanitize_risk("not a risk", 0))

    def test_ratings_categories_and_lists_are_cleaned(self):
        risk = sanitize_risk({
            "risk_title": "Tax law changes", "risk_summary": "Higher tax rates.", "risk_category": "Taxation",
            "likelihood": "probable", "impact": "elevated", "financial_transmission": ["revenue", "FX", "revenue"],
            "quantified_exposure": "N/A", "monitoring_signposts": "OECD Pillar Two",
            "evidence": [{"chunk_id": " c1 ", "quote": "x" * 400}, {"chunk_id": 3, "quote": "bad"}],
        }, 4)
        self.assertEqual(risk["risk_id"], "R5")
        self.assertEqual(risk["risk_category"], "Financial & Capital Structure")
        self.assertEqual((risk["likelihood"], risk["impact"], risk["severity"]), ("Medium", "High", "Severe"))
        self.assertEqual(risk["adjustments"], ["Likelihood not rated by the model; defaulted to Medium."])
        self.assertEqual(risk["financial_transmission"], ["Revenue"])
        self.assertIsNone(risk["quantified_exposure"])
        self.assertEqual(risk["monitoring_signposts"], ["OECD Pillar Two"])
        self.assertEqual(risk["evidence"], [{"chunk_id": "c1", "quote": "x" * 300}])

    def test_unknown_category_falls_back_to_keywords_then_operational(self):
        self.assertEqual(coerce_category("regulatory & legal", ""), "Regulatory & Legal")
        self.assertEqual(coerce_category(None, "Sole-source supplier for wafers"), "Supply Chain & Concentration")
        self.assertEqual(coerce_category("Other", "Key person dependence"), "Operational")


class TestVerifyInventory(unittest.TestCase):
    def test_a_quote_under_the_wrong_chunk_is_resolved(self):
        hit = verify_evidence(RiskEvidence(chunk_id="c2", quote=CUSTOMER_QUOTE), POOL)
        self.assertEqual((hit.chunk_id, hit.item, hit.verified), ("c1", "Item 1A", True))
        self.assertIsNone(verify_evidence(RiskEvidence(chunk_id="c1", quote="an invented sentence about power grids"), POOL))

    def test_a_risk_without_a_verified_quote_is_dropped_and_counted(self):
        kept = _risk("Customer concentration", [("c1", CUSTOMER_QUOTE), ("c1", "an invented quote about grids")])
        invented = _risk("Power grid", [("c1", "an invented sentence about power grids")])
        result = verify_inventory([kept, invented], POOL)
        self.assertEqual([r.risk_title for r in result.survivors], ["Customer concentration"])
        self.assertEqual((result.dropped_unverified, result.quotes_rejected), (1, 2))
        self.assertEqual(len(kept.evidence), 1)

    def test_unsupported_exposure_and_mitigation_are_removed(self):
        risk = _risk("Customer concentration", [("c1", CUSTOMER_QUOTE)], quantified_exposure="$2 billion",
                     mitigating_factors="Long-term contracts")
        verify_inventory([risk], POOL)
        self.assertIsNone(risk.quantified_exposure)
        self.assertIsNone(risk.mitigating_factors)
        self.assertEqual(risk.adjustments, [
            "Quantified exposure removed: its figures do not appear in the cited filing text.",
            "Mitigation removed: no verified filing quote supports it.",
        ])

    def test_legal_matter_only_outside_item_1a_is_capped_at_medium(self):
        risk = _risk("Patent lawsuit", [("c2", "We are party to a patent lawsuit in Texas")],
                     risk_category="Regulatory & Legal")
        verify_inventory([risk], POOL)
        self.assertEqual((risk.impact, risk.severity), ("Medium", "Moderate"))
        self.assertIn("legal matter sourced outside Item 1A", risk.adjustments[0])

    def test_a_stated_figure_keeps_a_high_impact(self):
        risk = _risk("Customer concentration", [("c1", CUSTOMER_QUOTE)], quantified_exposure="22% of total revenue")
        verify_inventory([risk], POOL)
        self.assertEqual((risk.impact, risk.severity, risk.adjustments), ("High", "Severe", []))


class TestMergeDuplicates(unittest.TestCase):
    def test_similar_titles_fold_into_the_higher_ranked_risk(self):
        top = _risk("Direct customer concentration", [("c1", CUSTOMER_QUOTE)])
        top.severity = "Severe"
        lower = _risk("Indirect customer concentration", [("c2", "patent lawsuit in Texas seeking damages")],
                      likelihood="Low", impact="Medium", monitoring_signposts=["Customer mix"])
        lower.severity = "Low"
        survivors, absorbed = merge_duplicates([lower, top])
        self.assertEqual(survivors, [top])
        self.assertIs(absorbed[id(lower)], top)
        self.assertEqual(len(top.evidence), 2)
        self.assertEqual(top.monitoring_signposts, ["Customer mix"])
        self.assertEqual(top.adjustments, ["Merged duplicate risk: 'Indirect customer concentration'."])

    def test_a_shared_chunk_lowers_the_title_bar(self):
        a = _risk("Google licensing revenue", [("c1", CUSTOMER_QUOTE)])
        b = _risk("Google antitrust ruling on licensing payments", [("c1", CUSTOMER_QUOTE)])
        c = _risk("Google antitrust ruling on licensing payments", [("c2", "patent lawsuit in Texas")])
        self.assertEqual(len(merge_duplicates([a, b])[0]), 1)
        self.assertEqual(len(merge_duplicates([_risk("Google licensing revenue", [("c1", CUSTOMER_QUOTE)]), c])[0]), 2)


if __name__ == "__main__":
    unittest.main()
