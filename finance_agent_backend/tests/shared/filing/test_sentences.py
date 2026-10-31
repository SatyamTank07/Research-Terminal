"""Sentence and word splitting shared by the Strategist, Risk and quote checks."""

import unittest

from app.agents.shared.filing.sentences import normalize_text, split_sentences, split_words


class TestNormalizeText(unittest.TestCase):
    def test_curly_quotes_dashes_and_spaces_become_plain_lowercase(self):
        self.assertEqual(normalize_text("The Company’s  Products — Chips Ltd "), "the company's products - chips ltd")

    def test_none_becomes_empty(self):
        self.assertEqual(normalize_text(None), "")


class TestSplitSentences(unittest.TestCase):
    def test_short_fragments_drop_and_bullets_split(self):
        text = ("Short one. We depend on a limited number of suppliers for key components used in our products. "
                "• Our largest customer accounted for 22% of total revenue in fiscal 2026 and may reduce orders.")
        self.assertEqual(split_sentences(text), [
            "We depend on a limited number of suppliers for key components used in our products.",
            "Our largest customer accounted for 22% of total revenue in fiscal 2026 and may reduce orders.",
        ])


class TestSplitWords(unittest.TestCase):
    def test_numbers_split_on_punctuation(self):
        self.assertEqual(split_words("Net sales rose 4.5% to $391,035 million"),
                         ["net", "sales", "rose", "4", "5", "to", "391", "035", "million"])

    def test_already_normalized_text_splits_the_same(self):
        text = "The Company’s products — chips"
        self.assertEqual(split_words(normalize_text(text)), split_words(text))
