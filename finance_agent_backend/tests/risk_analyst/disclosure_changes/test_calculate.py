"""Year-over-year Item 1A comparison: sentence classes, passages, preamble skipping, quote-level change labels."""

import unittest

from app.agents.specialized.risk_analyst.modules.disclosure_changes import (
    build_prior_index,
    classify_disclosure_change,
    diff_risk_disclosures,
    is_preamble,
    shingles,
)
from tests.risk_analyst.item1a_fixtures import item1a_chunk


class TestDisclosureDiff(unittest.TestCase):
    PRIOR = [
        item1a_chunk("p1", "Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries. "
                     "Global supply chains can be highly concentrated and an escalation of tensions could result in disruptions.", 0, fy=2024),
        item1a_chunk("p2", "We previously relied on a single legacy distribution partner for all of our retail channel shipments worldwide.", 1, fy=2024),
    ]
    CURRENT = [
        item1a_chunk("c1", "Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries. "
                     "Beginning in the second quarter, new tariffs were announced on imports into the United States from several countries.", 0, fy=2025),
    ]

    def test_new_unchanged_and_removed_sentences(self):
        diff = diff_risk_disclosures(self.CURRENT, self.PRIOR, 2024)
        self.assertEqual(diff.prior_fiscal_year, 2024)
        self.assertEqual(diff.current_sentences, 2)
        self.assertEqual(diff.unchanged_sentences, 1)
        self.assertEqual(diff.new_sentences, 1)
        self.assertGreaterEqual(diff.removed_sentences, 2)
        self.assertEqual(len(diff.new_passages), 1)
        self.assertEqual(diff.new_passages[0].chunk_id, "c1")
        self.assertIn("new tariffs", diff.new_passages[0].text)
        self.assertTrue(any("legacy distribution partner" in p.text for p in diff.removed_passages))
        self.assertAlmostEqual(diff.change_ratio_pct, 50.0)

    def test_identical_filings_have_no_changes(self):
        diff = diff_risk_disclosures(self.PRIOR, self.PRIOR, 2024)
        self.assertEqual(diff.new_sentences, 0)
        self.assertEqual(diff.removed_sentences, 0)
        self.assertEqual(diff.new_passages, [])

    def test_preamble_sentences_are_ignored(self):
        preamble = item1a_chunk("x", "The risks described below are not exhaustive and should be read with the forward-looking statements.", fy=2025)
        diff = diff_risk_disclosures([preamble], self.PRIOR, 2024)
        self.assertEqual(diff.current_sentences, 0)

    def test_quote_level_disclosure_change(self):
        index = build_prior_index(self.PRIOR)
        self.assertEqual(classify_disclosure_change(
            ["new tariffs were announced on imports into the United States from several countries"], index), "New")
        self.assertEqual(classify_disclosure_change(
            ["Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries."], index),
            "Unchanged")
        self.assertEqual(classify_disclosure_change(
            ["Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries.",
             "new tariffs were announced on imports into the United States from several countries"], index), "Expanded")
        self.assertEqual(classify_disclosure_change(["anything"], None), "Not Assessed")
        self.assertEqual(classify_disclosure_change([], index), "Not Assessed")


class TestSentenceTools(unittest.TestCase):
    def test_shingles_are_three_word_windows(self):
        self.assertEqual(shingles("Tariffs raise our costs"), {("tariffs", "raise", "our"), ("raise", "our", "costs")})
        self.assertEqual(shingles("Two words"), {("two", "words")})
        self.assertEqual(shingles(""), set())

    def test_introduction_boilerplate_is_preamble(self):
        self.assertTrue(is_preamble("The risks described below are not exhaustive."))
        self.assertFalse(is_preamble("We depend on a limited number of suppliers for our components."))


if __name__ == "__main__":
    unittest.main()
