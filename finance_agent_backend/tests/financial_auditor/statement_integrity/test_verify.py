"""Input corrections: a cited override is applied only to a known field of an existing record."""

import unittest

from app.agents.specialized.financial_auditor.context import build_statement_grids, extract_audit_inputs
from app.agents.specialized.financial_auditor.modules.statement_integrity import InputCorrection, apply_input_corrections
from tests.statement_fixtures import tsla_payloads


def _tsla_inputs():
    return extract_audit_inputs(build_statement_grids(tsla_payloads()), 2025, [2023, 2024, 2025]).to_tool_kwargs()


class TestApplyInputCorrections(unittest.TestCase):
    def test_a_cited_correction_replaces_the_value_and_keeps_the_original_untouched(self):
        inputs = _tsla_inputs()
        inputs["balance_sheet"]["marketable_securities"] = 0.0
        out = apply_input_corrections(inputs, [
            InputCorrection(target="balance_sheet", fiscal_year=2025, field="marketable_securities",
                            value=27546.0, reason="Short-term investments row", source_chunk_id="bs"),
        ])
        self.assertEqual(out["inputs"]["balance_sheet"]["marketable_securities"], 27546.0)
        self.assertEqual(inputs["balance_sheet"]["marketable_securities"], 0.0)
        self.assertEqual(out["applied"][0]["previous_value"], 0.0)

    def test_an_unknown_field_is_rejected(self):
        out = apply_input_corrections(_tsla_inputs(), [
            InputCorrection(target="balance_sheet", fiscal_year=2025, field="not_a_field", value=1.0, reason="x"),
        ])
        self.assertEqual(out["rejected"][0]["rejection"], "unknown field 'not_a_field'")

    def test_the_fiscal_year_itself_cannot_be_corrected(self):
        out = apply_input_corrections(_tsla_inputs(), [
            InputCorrection(target="annual", fiscal_year=2025, field="fiscal_year", value=2024.0, reason="x"),
        ])
        self.assertEqual(len(out["rejected"]), 1)

    def test_a_year_outside_the_window_is_rejected(self):
        out = apply_input_corrections(_tsla_inputs(), [
            InputCorrection(target="annual", fiscal_year=2019, field="revenue", value=1.0, reason="x"),
        ])
        self.assertEqual(out["rejected"][0]["rejection"], "no annual record for FY2019")

    def test_a_prior_balance_sheet_correction_must_name_its_year(self):
        out = apply_input_corrections(_tsla_inputs(), [
            InputCorrection(target="prior_balance_sheet", fiscal_year=2025, field="long_term_debt", value=1.0, reason="x"),
        ])
        self.assertEqual(out["rejected"][0]["rejection"], "no prior_balance_sheet record for FY2025")


if __name__ == "__main__":
    unittest.main()
