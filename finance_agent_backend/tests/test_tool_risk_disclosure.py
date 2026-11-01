"""Unit Tests for the deterministic Risk-Disclosure toolkit (Item 1A).

Validates the aggregate risk profile, materiality / boilerplate screens, the
quantified-disclosure checklist and the markdown renderers. No database or network access.
"""

import unittest

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.specialized.risk_analyst.modules.disclosure_changes.calculate import (
    derive_overall_profile,
    disclosure_covered,
    extract_quantified_disclosures,
    is_boilerplate,
    is_material_disclosure,
    render_risk_evidence_table,
    render_risk_matrix_markdown,
    states_immaterial,
    substantiates_high_impact,
)

CONCENTRATION = (
    "We receive a significant amount of our revenue from a limited number of customers. "
    "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
    "another direct customer represented 14% of total revenue, all of which were primarily attributable "
    "to the Compute & Networking segment."
)


def _chunk(chunk_id: str, content: str, idx: int = 0, item: str = "Item 1A", fy: int = 2026) -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        ticker="TEST",
        fiscal_year=fy,
        item=item,
        breadcrumb=f"TEST > 10-K FY{fy} > PART I > {item}: Risk Factors",
        sub_section="Risk Factors",
        content=content,
        chunk_index=idx,
    )


class TestOverallProfile(unittest.TestCase):
    def test_overall_profile(self):
        self.assertEqual(derive_overall_profile(["Severe", "Severe", "Low"]), "High")
        self.assertEqual(derive_overall_profile(["Severe", "Low", "Low"]), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate"] * 3), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate", "Low"]), "Low")
        self.assertEqual(derive_overall_profile([]), "Not Assessed")


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


class TestQuantifiedDisclosures(unittest.TestCase):
    def test_extracts_figures_about_revenue_charges_and_trade(self):
        chunks = [
            _chunk("q1", CONCENTRATION, 0),
            _chunk("q2", "We incurred a $4.5 billion charge associated with excess inventory and purchase obligations. "
                         "Our employees are located in more than 30 countries around the world today.", 1),
        ]
        found = extract_quantified_disclosures(chunks)
        texts = [d.text for d in found]
        self.assertEqual([d.chunk_id for d in found], ["q1", "q2"])
        self.assertTrue(any("22% of total revenue" in t for t in texts))
        self.assertTrue(any("$4.5 billion charge" in t for t in texts))
        self.assertFalse(any("30 countries" in t for t in texts))  # a figure, but not an exposure topic
        self.assertEqual(len(extract_quantified_disclosures(chunks, max_items=1)), 1)

    def test_material_disclosures_exclude_statutory_maximums(self):
        self.assertTrue(is_material_disclosure(CONCENTRATION))
        self.assertTrue(is_material_disclosure("We incurred a $4.5 billion charge for excess inventory."))
        self.assertFalse(is_material_disclosure(
            "We could be subject to penalties of up to €20 million or 4% of worldwide revenue, whichever is greater."))
        self.assertFalse(is_material_disclosure("Our employees work in more than 30 countries."))

    def test_coverage_by_partial_quote(self):
        sentence = ("For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
                    "another direct customer represented 14% of total revenue.")
        self.assertTrue(disclosure_covered(sentence, ["sales to one direct customer represented 22% of total revenue"]))
        self.assertFalse(disclosure_covered(sentence, ["We rely on third-party foundries located in Taiwan."]))
        self.assertFalse(disclosure_covered(sentence, []))

    def test_high_impact_substantiation(self):
        self.assertTrue(substantiates_high_impact(["anything"], "22% of revenue"))
        self.assertTrue(substantiates_high_impact(["sales to one customer represented 22% of total revenue"], None))
        self.assertTrue(substantiates_high_impact(["We incurred significant costs to remediate the defect"], None))
        self.assertTrue(substantiates_high_impact(["Export restrictions have adversely affected our China revenue"], None))
        self.assertFalse(substantiates_high_impact(["A cyberattack could disrupt our operations and may harm results"], None))
        self.assertFalse(substantiates_high_impact([], None))


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
