"""Statement integrity: identity checks, completeness, usability, restatement notes and data-quality findings."""

import unittest

from app.agents.specialized.financial_auditor.context import build_statement_grids, extract_audit_inputs
from app.agents.specialized.financial_auditor.modules.statement_integrity import (
    assess_statement_integrity,
    flag_data_quality,
    note_restatements,
    record_data_quality,
    summarize_for_judge,
)
from tests.statement_fixtures import TSLA_FY2025_BALANCE_SHEET, TSLA_FY2025_INCOME_STATEMENT, tsla_payloads


def _extract(payloads):
    return extract_audit_inputs(build_statement_grids(payloads), 2025, [2023, 2024, 2025])


def _broken_balance_sheet():
    payloads = tsla_payloads()
    payloads["balance_sheet"][0]["table_markdown"] = TSLA_FY2025_BALANCE_SHEET.replace(
        "| Total liabilities and equity | $ | 137,806 |", "| Total liabilities and equity | $ | 147,806 |"
    )
    return _extract(payloads)


class TestStatementIntegrity(unittest.TestCase):
    def test_clean_statements_pass_every_identity_and_are_usable(self):
        integrity = assess_statement_integrity(_extract(tsla_payloads()))
        self.assertTrue(integrity.identity_checks)
        self.assertEqual(integrity.failed_identity_checks, [])
        self.assertEqual(integrity.completeness_score, 1.0)
        self.assertTrue(integrity.is_usable)

    def test_a_balance_sheet_that_does_not_balance_blocks_the_deterministic_audit(self):
        integrity = assess_statement_integrity(_broken_balance_sheet())
        self.assertTrue(any(c["critical"] for c in integrity.failed_identity_checks))
        self.assertFalse(integrity.is_usable)

    def test_a_missing_cash_flow_statement_is_not_usable(self):
        payloads = tsla_payloads()
        payloads["cash_flow"] = []
        integrity = assess_statement_integrity(_extract(payloads))
        self.assertFalse(integrity.is_usable)
        self.assertLess(integrity.completeness_score, 1.0)

    def test_the_judge_sees_only_the_failed_checks(self):
        extraction = _broken_balance_sheet()
        integrity = assess_statement_integrity(extraction)
        summary = summarize_for_judge(extraction, integrity)
        self.assertEqual(list(summary), ["years", "completeness_score", "missing_fields", "failed_identity_checks",
                                         "cross_filing_differences", "warnings"])
        self.assertEqual(summary["failed_identity_checks"], integrity.failed_identity_checks)


class TestRestatementNotes(unittest.TestCase):
    def test_a_single_filing_cannot_be_reviewed_for_restatements(self):
        self.assertEqual(note_restatements(_extract(tsla_payloads()), tsla_payloads()),
                         ["Single filing available; cross-filing restatement review not possible."])

    def test_a_restated_figure_names_both_filings_and_keeps_the_latest(self):
        payloads = tsla_payloads()
        older = TSLA_FY2025_INCOME_STATEMENT.replace("| 2025 | Col_3 | 2024 | Col_5 | 2023 |",
                                                     "| 2024 | Col_3 | 2023 | Col_5 | 2022 |")
        older = older.replace("| Total revenues | 94,827 |  |  | 97,690 |", "| Total revenues | 97,000 |  |  | 97,690 |")
        payloads["income_statement"].append({"chunk_id": "old", "fiscal_year": 2024,
                                             "statement_type": "income_statement", "table_markdown": older})
        notes = note_restatements(_extract(payloads), payloads)
        self.assertTrue(any(n.startswith("FY2024 revenue (income_statement): 97,000.0 in the FY2024 10-K")
                            and n.endswith("latest filing figure used.") for n in notes))

    def test_agreeing_filings_say_no_restatement_was_found(self):
        tables = {"income_statement": [{"fiscal_year": 2025}, {"fiscal_year": 2024}]}
        notes = note_restatements(_extract(tsla_payloads()), tables)
        self.assertTrue(notes[0].startswith("No retrospective restatements identified"))
        self.assertIn("FY2025, FY2024", notes[0])


class TestDataQuality(unittest.TestCase):
    def test_no_statements_means_llm_transcribed_inputs(self):
        dq = record_data_quality(None, None, "llm_fallback", ["log"], True)
        self.assertIsNone(dq.completeness_score)
        self.assertEqual(dq.warnings, ["Primary statements could not be retrieved or parsed; inputs transcribed by the LLM."])
        self.assertTrue(dq.summary_may_be_stale)

    def test_a_failed_critical_identity_is_a_high_data_quality_finding(self):
        extraction = _broken_balance_sheet()
        dq = record_data_quality(extraction, assess_statement_integrity(extraction), "deterministic", [], False)
        findings = flag_data_quality(dq)
        self.assertEqual({(f.code, f.severity) for f in findings}, {("DATA_QUALITY", "high")})
        self.assertIn("Identity check failed", findings[0].message)

    def test_llm_transcription_is_flagged_for_verification(self):
        dq = record_data_quality(None, None, "llm_fallback", [], False)
        self.assertEqual([(f.severity, f.metric) for f in flag_data_quality(dq)], [("info", "extraction_mode")])

    def test_clean_deterministic_statements_raise_no_data_quality_finding(self):
        extraction = _extract(tsla_payloads())
        dq = record_data_quality(extraction, assess_statement_integrity(extraction), "deterministic", [], False)
        self.assertEqual(flag_data_quality(dq), [])


if __name__ == "__main__":
    unittest.main()
