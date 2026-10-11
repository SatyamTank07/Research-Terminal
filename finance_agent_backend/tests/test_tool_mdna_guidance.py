"""Tests for the deterministic Item 7 guidance screen."""

import unittest

from app.agents.tools.mdna_guidance import (
    assess_mdna_guidance,
    is_outlook_sentence,
    is_quantified_guidance,
    margin_evidence_chunk_ids,
    split_sentences,
)
from tests.forecast_fixtures import (
    AMZN_FX_SENTENCE,
    AMZN_GUIDANCE_CHUNK,
    AMZN_SAFE_HARBOR_SENTENCE,
    BOILERPLATE_CHUNK,
    GUIDANCE_CHUNK,
    HISTORY_CHUNK,
    MARGIN_OUTLOOK_CHUNK,
    make_chunk,
)


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

    def test_amzn_safe_harbor_sentence_is_not_guidance(self):
        self.assertFalse(is_outlook_sentence(AMZN_SAFE_HARBOR_SENTENCE))
        self.assertFalse(is_quantified_guidance(AMZN_SAFE_HARBOR_SENTENCE))

    def test_amzn_fx_impact_note_is_not_quantified_guidance(self):
        self.assertTrue(is_outlook_sentence(AMZN_FX_SENTENCE))  # forward-looking ...
        self.assertFalse(is_quantified_guidance(AMZN_FX_SENTENCE))  # ... but FX-only, not an operating driver

    def test_amzn_guidance_section_is_not_informative(self):
        # The closing quote after "Risk Factors." used to glue the hedging sentence onto the quarterly bullet.
        sentences = split_sentences(AMZN_GUIDANCE_CHUNK)
        self.assertIn(AMZN_SAFE_HARBOR_SENTENCE, sentences)
        self.assertIn(AMZN_FX_SENTENCE, sentences)
        a = assess_mdna_guidance([make_chunk("amzn", AMZN_GUIDANCE_CHUNK)])
        self.assertEqual(a.quantified_guidance, [])
        self.assertFalse(a.is_informative)

    def test_single_quarter_guidance_is_excluded(self):
        for text in (
            "Net sales are expected to be between $173.5 billion and $178.5 billion in the first quarter of 2026.",
            "For Q1 we expect revenue of $40 billion, plus or minus 2%.",
            "Operating income is expected to be between $16.5 billion and $21.5 billion for the quarter ending March 31.",
        ):
            self.assertFalse(is_quantified_guidance(text), text)

    def test_annual_operating_guidance_still_counts(self):
        for text in (
            "We expect capital expenditures of approximately $14 billion in fiscal 2026.",
            "We currently expect our capital expenditures to be in excess of $20 billion in 2026, driven by AI.",
            "We expect full-year revenue growth of 8% to 10% in 2026.",
            "Over the next three years we expect gross margin to expand by 200 basis points.",
        ):
            self.assertTrue(is_quantified_guidance(text), text)

    def test_non_operating_figures_are_not_guidance(self):
        for text in (
            "We expect interest income of $2.0 billion in 2026.",
            "We expect our effective tax rate to be approximately 16% in 2026.",
            "We plan to repurchase $10 billion of shares in 2026.",
        ):
            self.assertFalse(is_quantified_guidance(text), text)

    def test_margin_evidence_chunks_need_forward_looking_margin_language(self):
        chunks = [
            make_chunk("m1", MARGIN_OUTLOOK_CHUNK),
            make_chunk("h1", "Operating margin was 11.2% in 2025, up from 10.8%."),  # historical only
            make_chunk("g1", GUIDANCE_CHUNK),  # forward-looking, but CapEx not margins
            make_chunk("b1", AMZN_GUIDANCE_CHUNK),  # boilerplate / quarterly
        ]
        self.assertEqual(margin_evidence_chunk_ids(chunks), {"m1"})
        self.assertEqual(margin_evidence_chunk_ids([]), set())

    def test_none_content_is_safe(self):
        chunk = make_chunk("n1", "")
        chunk.content = None
        self.assertEqual(assess_mdna_guidance([chunk]).forward_looking_count, 0)
        self.assertEqual(margin_evidence_chunk_ids([chunk]), set())


if __name__ == "__main__":
    unittest.main()
