"""Write-step rules: what the prose screen removes and how every prose field is backed up."""

import unittest

from app.agents.specialized.lead_synthesizer.write import (
    PROSE_FIELDS,
    _YOY_CLAIM,
    drop_sentences,
    fill_empty_prose,
    screen_prose,
    strip_layout,
    write_fallback_prose,
)
from tests.test_agent_lead_synthesizer import (
    _create_mock_audit,
    _create_mock_dcf,
    _create_mock_forecast,
    _create_mock_moat,
    _create_mock_risk,
)

YOY_TEXT = "Supply risk rose year-over-year. Demand stays strong."


class TestProseScreen(unittest.TestCase):
    def test_dropped_sentences_keep_the_bullet_prefix(self):
        text = "- Margins held. Compared to the prior year it improved.\n- Cash is ample."
        self.assertEqual(drop_sentences(text, _YOY_CLAIM), "- Margins held.\n- Cash is ample.")

    def test_headings_tables_and_code_fences_are_stripped_from_prose(self):
        text = "## Heading\nThe moat holds.\n| a | b |\n```\nCash is ample."
        self.assertEqual(strip_layout(text), "The moat holds.\n\nCash is ample.")

    def test_year_over_year_risk_claims_are_removed_only_without_a_baseline(self):
        raw = {"risk_commentary": YOY_TEXT}
        with_baseline, _ = screen_prose(raw, "", _create_mock_risk())
        no_baseline, _ = screen_prose(raw, "", _create_mock_risk().model_copy(update={"disclosure_changes": None}))
        self.assertEqual(with_baseline["risk_commentary"], YOY_TEXT)
        self.assertEqual(no_baseline["risk_commentary"], "Demand stays strong.")

    def test_sentences_with_figures_absent_from_the_context_are_removed_and_counted(self):
        raw = {"valuation_commentary": "Fair value is $123.45 per share. The moat is wide."}
        prose, removed = screen_prose(raw, "context with no such figure", None)
        self.assertEqual((prose["valuation_commentary"], removed), ("The moat is wide.", 1))

    def test_an_invented_report_date_is_removed(self):
        prose, _ = screen_prose({"executive_summary": "The report date is today. The moat is wide."}, "", None)
        self.assertEqual(prose["executive_summary"], "The moat is wide.")


class TestProseFallback(unittest.TestCase):
    def test_every_field_the_llm_left_empty_takes_its_fallback(self):
        fallback = write_fallback_prose(
            "AAPL", "Apple Inc.", 2025, business_moat=_create_mock_moat(), financial_audit=_create_mock_audit(),
            forecast=_create_mock_forecast(), dcf_valuation=_create_mock_dcf(), risk_audit=_create_mock_risk(),
        )
        prose = fill_empty_prose({"executive_summary": "LLM summary.", "bottom_line": None}, fallback)
        self.assertEqual(set(prose), set(PROSE_FIELDS))
        self.assertEqual(prose["executive_summary"], "LLM summary.")
        self.assertEqual(prose["bottom_line"], fallback["bottom_line"])
        for key in ("pillar_1_business_moat", "pillar_2_financial_durability", "pillar_3_valuation_asymmetry"):
            self.assertTrue(prose[key])

    def test_a_missing_analysis_is_stated_as_unavailable(self):
        fallback = write_fallback_prose("AAPL", "Apple Inc.", 2025, None, None, None, None, None)
        self.assertEqual(fallback["pillar_3_valuation_asymmetry"], "The DCF valuation is unavailable for this report.")
        self.assertEqual(fallback["executive_summary"], "Apple Inc. (AAPL), FY2025 Form 10-K.")
