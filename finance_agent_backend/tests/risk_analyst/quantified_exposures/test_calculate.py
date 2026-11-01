"""Quantified-disclosure checklist: stated figures about exposures, materiality, coverage by quotes, repair targets."""

import unittest

from app.agents.specialized.risk_analyst.modules.quantified_exposures import (
    QuantifiedDisclosure,
    check_quantified_coverage,
    disclosure_covered,
    extract_quantified_disclosures,
    is_material_disclosure,
    repair_targets,
)
from tests.risk_analyst.item1a_fixtures import CONCENTRATION, item1a_chunk

STATUTORY = "We could be subject to penalties of up to €20 million or 4% of worldwide revenue, whichever is greater."


class TestQuantifiedDisclosures(unittest.TestCase):
    def test_extracts_figures_about_revenue_charges_and_trade(self):
        chunks = [
            item1a_chunk("q1", CONCENTRATION, 0),
            item1a_chunk("q2", "We incurred a $4.5 billion charge associated with excess inventory and purchase obligations. "
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
        self.assertFalse(is_material_disclosure(STATUTORY))
        self.assertFalse(is_material_disclosure("Our employees work in more than 30 countries."))

    def test_coverage_by_partial_quote(self):
        sentence = ("For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
                    "another direct customer represented 14% of total revenue.")
        self.assertTrue(disclosure_covered(sentence, ["sales to one direct customer represented 22% of total revenue"]))
        self.assertFalse(disclosure_covered(sentence, ["We rely on third-party foundries located in Taiwan."]))
        self.assertFalse(disclosure_covered(sentence, []))


class TestCoverageCheck(unittest.TestCase):
    CHECKLIST = [
        QuantifiedDisclosure(chunk_id="q1", text="Sales to one direct customer represented 22% of total revenue."),
        QuantifiedDisclosure(chunk_id="q2", text="We incurred a $4.5 billion charge for excess inventory."),
        QuantifiedDisclosure(chunk_id="q3", text=STATUTORY),
    ]

    def test_cited_figures_are_covered_and_the_rest_noted(self):
        coverage = check_quantified_coverage(self.CHECKLIST, ["one direct customer represented 22% of total revenue"])
        self.assertEqual([d.chunk_id for d in coverage.uncovered], ["q2", "q3"])
        self.assertEqual(coverage.listed, (self.CHECKLIST[1].text, STATUTORY))
        self.assertEqual(coverage.note, "2 of 3 quantified Item 1A disclosures are not cited by any verified risk.")

    def test_full_coverage_needs_no_note(self):
        coverage = check_quantified_coverage(self.CHECKLIST[:1], ["one direct customer represented 22% of total revenue"])
        self.assertEqual((coverage.uncovered, coverage.listed, coverage.note), ((), (), None))

    def test_data_quality_lists_at_most_five_uncited_figures(self):
        checklist = [QuantifiedDisclosure(chunk_id=f"q{i}", text=f"Customer {i} represented {10 + i}% of revenue.")
                     for i in range(7)]
        coverage = check_quantified_coverage(checklist, [])
        self.assertEqual(len(coverage.uncovered), 7)
        self.assertEqual(len(coverage.listed), 5)

    def test_repair_asks_only_about_material_figures_at_most_four(self):
        self.assertEqual([d.chunk_id for d in repair_targets(self.CHECKLIST)], ["q1", "q2"])
        many = [QuantifiedDisclosure(chunk_id=f"m{i}", text=f"Customer {i} represented {10 + i}% of revenue.")
                for i in range(6)]
        self.assertEqual([d.chunk_id for d in repair_targets(many)], ["m0", "m1", "m2", "m3"])


if __name__ == "__main__":
    unittest.main()
