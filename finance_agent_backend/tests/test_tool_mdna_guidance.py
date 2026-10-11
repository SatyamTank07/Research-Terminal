"""Tests for the deterministic Item 7 guidance screen."""

import unittest

from app.agents.tools.mdna_guidance import assess_mdna_guidance
from tests.forecast_fixtures import BOILERPLATE_CHUNK, GUIDANCE_CHUNK, HISTORY_CHUNK, make_chunk


class TestMdnaGuidance(unittest.TestCase):
    def test_guidance_chunk_is_informative(self):
        a = assess_mdna_guidance([make_chunk("c1", GUIDANCE_CHUNK), make_chunk("c2", HISTORY_CHUNK)])
        self.assertEqual(a.chunk_count, 2)
        self.assertEqual(a.forward_looking_count, 2)
        self.assertTrue(a.is_informative)
        self.assertEqual(len(a.quantified_guidance), 1)
        self.assertEqual(a.quantified_guidance[0].chunk_id, "c1")
        self.assertTrue(a.quantified_guidance[0].sentence.startswith("We expect capital expenditures"))

    def test_safe_harbor_boilerplate_is_not_guidance(self):
        a = assess_mdna_guidance([make_chunk("b1", BOILERPLATE_CHUNK)])
        self.assertEqual(a.forward_looking_count, 0)
        self.assertFalse(a.is_informative)

    def test_historical_narrative_is_not_guidance(self):
        self.assertEqual(assess_mdna_guidance([make_chunk("c2", HISTORY_CHUNK)]).forward_looking_count, 0)

    def test_accounting_guidance_is_not_outlook(self):
        a = assess_mdna_guidance([make_chunk("c3", "We adopted new accounting guidance on leases in 2025.")])
        self.assertEqual(a.forward_looking_count, 0)

    def test_duplicate_sentences_count_once(self):
        a = assess_mdna_guidance([make_chunk("c1", GUIDANCE_CHUNK), make_chunk("c9", GUIDANCE_CHUNK)])
        self.assertEqual(a.forward_looking_count, 2)

    def test_disclosure_boilerplate_is_not_guidance(self):
        text = (
            "Unrecognized compensation cost of $9.5 billion is expected to be recognized over 2.6 years. "
            "We believe our cash of $65.2 billion will be sufficient to meet our needs over the next 12 months. "
            "A hypothetical 10% change in exchange rates would not be expected to have a material effect. "
            "Remaining performance obligations of $12.0 billion are expected to be recognized over the next year. "
            "We expect to contribute $50 million to our pension plans in 2026. "
            "In 2025 the FASB issued guidance on income tax disclosures that we expect to adopt. "
            "Moody's affirmed our rating with a stable outlook."
        )
        a = assess_mdna_guidance([make_chunk("x1", text)])
        self.assertEqual(a.quantified_guidance, [])
        self.assertFalse(a.is_informative)

    def test_qualitative_only_outlook_is_not_informative(self):
        text = "We expect demand to remain strong. We anticipate continued investment in our products."
        a = assess_mdna_guidance([make_chunk("q1", text)])
        self.assertEqual(a.forward_looking_count, 2)
        self.assertFalse(a.is_informative)

    def test_no_chunks(self):
        a = assess_mdna_guidance([])
        self.assertEqual((a.chunk_count, a.is_informative), (0, False))


if __name__ == "__main__":
    unittest.main()
