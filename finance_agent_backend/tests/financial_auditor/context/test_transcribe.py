"""llm_fallback reconciliation: parsed statements fill what the LLM's transcription left out, never overwrite it."""

import unittest

from app.agents.specialized.financial_auditor.context import build_statement_grids, extract_audit_inputs
from app.agents.specialized.financial_auditor.context.transcribe import gap_fill_transcribed_inputs
from tests.statement_fixtures import tsla_payloads


def _tsla_extraction():
    return extract_audit_inputs(build_statement_grids(tsla_payloads()), 2025, [2023, 2024, 2025])


def _llm_args(**balance_overrides):
    balance = {"fiscal_year": 2025, "cash_and_equivalents": 16513.0, "marketable_securities": 0.0,
               "short_term_debt": 1640.0, "long_term_debt": 6736.0, "stockholders_equity": 82137.0,
               "weighted_diluted_shares": 3528.0, **balance_overrides}
    return {
        "annual_financials": [
            {"fiscal_year": 2025, "revenue": 94827.0, "operating_income": 4355.0, "net_income": 3794.0,
             "operating_cash_flow": 14747.0, "capital_expenditures": 8527.0},
        ],
        "balance_sheet": balance,
        "normalization_adjustments": [{"fiscal_year": 2025, "label": "x", "amount": 1.0,
                                       "direction": "inflated_reported_earnings"}],
    }


class TestGapFillTranscribedInputs(unittest.TestCase):
    def test_a_zero_securities_line_is_filled_from_the_parsed_balance_sheet(self):
        filled = gap_fill_transcribed_inputs(_llm_args(), _tsla_extraction())
        self.assertTrue(filled["changed"])
        self.assertEqual(filled["inputs"]["balance_sheet"]["marketable_securities"], 27546.0)
        self.assertTrue(any("Gap-filled FY2025.balance_sheet.marketable_securities" in m for m in filled["log"]))

    def test_omitted_line_items_are_added_and_the_prior_balance_sheet_is_attached(self):
        filled = gap_fill_transcribed_inputs(_llm_args(), _tsla_extraction())
        self.assertEqual(filled["inputs"]["annual_financials"][0]["gross_profit"], 17094.0)
        self.assertEqual(filled["inputs"]["prior_balance_sheet"]["fiscal_year"], 2024)
        self.assertIn("Added FY2024 prior balance sheet from parsed statements.", filled["log"])

    def test_a_disagreeing_llm_value_is_kept_and_logged(self):
        filled = gap_fill_transcribed_inputs(_llm_args(cash_and_equivalents=20000.0), _tsla_extraction())
        self.assertEqual(filled["inputs"]["balance_sheet"]["cash_and_equivalents"], 20000.0)
        self.assertTrue(any(m.startswith("Disagreement on FY2025.balance_sheet.cash_and_equivalents") for m in filled["log"]))

    def test_the_llm_normalization_adjustments_pass_through(self):
        filled = gap_fill_transcribed_inputs(_llm_args(), _tsla_extraction())
        self.assertEqual(filled["adjustments"][0]["label"], "x")

    def test_invalid_llm_inputs_skip_the_gap_fill(self):
        args = _llm_args()
        del args["balance_sheet"]["stockholders_equity"]
        filled = gap_fill_transcribed_inputs(args, _tsla_extraction())
        self.assertFalse(filled["changed"])
        self.assertIsNone(filled["inputs"])
        self.assertTrue(filled["log"][0].startswith("Gap-fill skipped: LLM inputs invalid"))


if __name__ == "__main__":
    unittest.main()
