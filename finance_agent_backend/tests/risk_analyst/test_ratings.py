"""The rating scale: free-form ratings normalized, levels capped and compared, the severity matrix, the rank order."""

import unittest
from types import SimpleNamespace

from app.agents.specialized.risk_analyst.ratings import (
    cap_level,
    derive_severity,
    higher_level,
    normalize_level,
    rank_key,
)


class TestSeverityMatrix(unittest.TestCase):
    def test_severity_matrix(self):
        self.assertEqual(derive_severity("High", "High"), "Severe")
        self.assertEqual(derive_severity("Medium", "High"), "Severe")
        self.assertEqual(derive_severity("Low", "High"), "Moderate")
        self.assertEqual(derive_severity("High", "Medium"), "Moderate")
        self.assertEqual(derive_severity("Low", "Medium"), "Low")
        self.assertEqual(derive_severity("High", "Low"), "Low")

    def test_an_unrated_pair_is_moderate(self):
        self.assertEqual(derive_severity("Unknown", "High"), "Moderate")


class TestLevels(unittest.TestCase):
    def test_level_normalization_and_caps(self):
        self.assertEqual(normalize_level("severe"), "High")
        self.assertEqual(normalize_level(" Moderate "), "Medium")
        self.assertEqual(normalize_level("low"), "Low")
        self.assertIsNone(normalize_level("unknown"))
        self.assertIsNone(normalize_level(3))
        self.assertEqual(cap_level("High", "Medium"), "Medium")
        self.assertEqual(cap_level("Low", "Medium"), "Low")

    def test_higher_level_keeps_the_more_serious_rating(self):
        self.assertEqual(higher_level("Low", "High"), "High")
        self.assertEqual(higher_level("Medium", "Low"), "Medium")
        self.assertEqual(higher_level("Medium", "Medium"), "Medium")


class TestRankOrder(unittest.TestCase):
    def test_severity_then_impact_then_likelihood(self):
        risks = [
            SimpleNamespace(name="low", severity="Low", impact="Low", likelihood="High"),
            SimpleNamespace(name="moderate_high_impact", severity="Moderate", impact="High", likelihood="Low"),
            SimpleNamespace(name="moderate_likely", severity="Moderate", impact="Medium", likelihood="High"),
            SimpleNamespace(name="severe", severity="Severe", impact="High", likelihood="Medium"),
        ]
        self.assertEqual([r.name for r in sorted(risks, key=rank_key)],
                         ["severe", "moderate_high_impact", "moderate_likely", "low"])


if __name__ == "__main__":
    unittest.main()
