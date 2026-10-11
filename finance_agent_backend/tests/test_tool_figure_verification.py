"""Figure verification guard for model-written prose (app/agents/tools/figure_verification.py)."""

import unittest

from app.agents.tools.figure_verification import (
    collect_known_figures,
    sentence_is_verified,
    strip_unverified_sentences,
)

CONTEXT = (
    "Revenue | 716,924.0 | Operating margin % | 11.16% | Base-Case Fair Value Per Share: $66.74 | "
    "Current price $262.43 | Upside -74.6% | WACC 11.64% | Net Debt: $-57,381.0M | implied terminal EV/EBITDA 3.65x | "
    "growth 12.38% | 2.2B+ active devices"
)


class TestFigureVerification(unittest.TestCase):
    def setUp(self):
        self.known = collect_known_figures(CONTEXT)

    def test_rounded_and_rescaled_figures_match(self):
        for sentence in (
            "Revenue was $716.9 billion.",
            "Revenue was $717 billion.",
            "Revenue was $716,924M.",
            "The margin is 11.2%.",
            "Fair value is $66.74, about $67 per share.",
            "The stock trades at $262.43, implying -74.6% downside.",
            "Net cash is $57.4 billion.",
            "Terminal multiple of 3.65x.",
            "Growth was roughly 12%.",
            "An installed base of 2.2B+ devices.",
        ):
            with self.subTest(sentence=sentence):
                self.assertTrue(sentence_is_verified(sentence, self.known))

    def test_untraceable_figures_fail(self):
        for sentence in (
            "Revenue will reach $1,250 billion.",
            "The margin expands to 18.5%.",
            "Fair value is $310.00 per share.",
            "The multiple is 22x.",
        ):
            with self.subTest(sentence=sentence):
                self.assertFalse(sentence_is_verified(sentence, self.known))

    def test_years_counts_and_identifiers_are_not_figures(self):
        for sentence in (
            "In FY2025 the 10-K lists 5 risks under Item 1A.",
            "Risk R2 is one of 3 Severe risks over a 10-year horizon.",
            "By 2030 the company plans further investment.",
        ):
            with self.subTest(sentence=sentence):
                self.assertTrue(sentence_is_verified(sentence, self.known))

    def test_strip_keeps_bullets_and_counts_removals(self):
        text = (
            "- Revenue was $716.9 billion. Management targets $2 trillion.\n"
            "- The margin is 11.16%.\n"
            "- Fair value of $500.00."
        )
        cleaned, removed = strip_unverified_sentences(text, self.known)
        self.assertEqual(cleaned, "- Revenue was $716.9 billion.\n- The margin is 11.16%.")
        self.assertEqual(removed, 2)

    def test_empty_inputs(self):
        self.assertEqual(strip_unverified_sentences("", self.known), ("", 0))
        self.assertEqual(collect_known_figures(""), [])
        self.assertFalse(sentence_is_verified("Fair value is $10.50.", []))


if __name__ == "__main__":
    unittest.main()
