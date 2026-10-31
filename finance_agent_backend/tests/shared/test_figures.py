"""Figure verification guard for model-written prose (app/agents/shared/figures.py)."""

import unittest

from app.agents.shared.figures import (
    annotate_sentences,
    attribute_sentence,
    build_figure_index,
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


class TestFigureAttribution(unittest.TestCase):
    def setUp(self):
        self.index = build_figure_index([
            ("416161.000000", "is-2025"),
            ("31.970000%", "is-2025"),
            ("98767.000000", "cf-2025"),
            ("416161.000000", "segment"),
            ("Net sales increased 6% to $416.2 billion", "quote"),
        ])

    def test_best_source_per_figure(self):
        self.assertEqual(attribute_sentence("Revenue was $416.2 billion.", self.index), ["is-2025"])

    def test_figures_in_sentence_order_deduped(self):
        s = "Margin was 31.97% on revenue of $416,161M while FCF was $98.8 billion."
        self.assertEqual(attribute_sentence(s, self.index), ["is-2025", "cf-2025"])

    def test_years_small_integers_and_unknown_figures_are_ignored(self):
        self.assertEqual(attribute_sentence("In FY2025 the 3 segments earned $123,456M.", self.index), [])

    def test_limit(self):
        index = build_figure_index([("1000", "a"), ("2000", "b"), ("3000", "c"), ("4000", "d")])
        self.assertEqual(attribute_sentence("$1,000M, $2,000M, $3,000M and $4,000M.", index), ["a", "b", "c"])

    def test_annotate_inserts_before_final_punctuation_and_keeps_bullets(self):
        text = "- Revenue was $416.2 billion. No figures here.\n- FCF was **$98.8 billion.**"
        out = annotate_sentences(text, self.index, lambda keys: "".join(f"[{k}]" for k in keys))
        self.assertEqual(
            out, "- Revenue was $416.2 billion[is-2025]. No figures here.\n- FCF was **$98.8 billion[cf-2025].**"
        )

    def test_annotate_without_terminal_punctuation_appends(self):
        self.assertEqual(
            annotate_sentences("Revenue was $416.2 billion", self.index, lambda keys: "[x]"),
            "Revenue was $416.2 billion[x]",
        )

    def test_annotate_leaves_unattributed_text_unchanged(self):
        self.assertEqual(annotate_sentences("", self.index, lambda keys: "[x]"), "")
        self.assertEqual(annotate_sentences("No numbers. None at all.", self.index, lambda keys: "[x]"), "No numbers. None at all.")


class TestAttributionUnitsAndBlockers(unittest.TestCase):
    def test_percent_figures_match_only_percent_entries(self):
        index = build_figure_index([("25.000000", "amount-row"), ("6.400000%", "growth-row")])
        self.assertEqual(attribute_sentence("The base case implies 25% upside.", index), [])
        self.assertEqual(attribute_sentence("Revenue grew 6.4%.", index), ["growth-row"])
        self.assertEqual(attribute_sentence("Capex was $25M.", index), ["amount-row"])
        self.assertEqual(attribute_sentence("Capex was $6.4M.", index), [])

    def test_unciteable_figure_blocks_lower_priority_sources(self):
        index = build_figure_index([("3.000000%", None), ("3.000000%", "statement"), ("7.000000%", "statement")])
        self.assertEqual(attribute_sentence("Terminal growth of 3% vs margin growth of 7%.", index), ["statement"])
        self.assertEqual(annotate_sentences("Terminal growth of 3%.", index, lambda keys: "[x]"), "Terminal growth of 3%.")
