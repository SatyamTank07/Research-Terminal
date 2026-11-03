"""filing_resolution: explicit year, else the session year on a follow-up, else the latest; substitution flagged."""

import unittest

from app.agents.specialized.supervisor.modules.filing_resolution import (
    FilingChoice,
    RequestedYear,
    choose_filing_year,
    explain_missing_filing,
    extract_fiscal_year,
    names_entity,
    resolve_requested_year,
)

SESSION = {"active_ticker": "AAPL", "active_company": "Apple Inc.", "active_fiscal_year": 2024}
AVAILABLE = [2025, 2024]


class TestExtractFiscalYear(unittest.TestCase):
    def test_named_fiscal_years(self):
        self.assertEqual(extract_fiscal_year("Analyze FY2023 results"), 2023)
        self.assertEqual(extract_fiscal_year("fiscal year 2024 audit"), 2024)
        self.assertEqual(extract_fiscal_year("FY'24 margins"), 2024)
        self.assertEqual(extract_fiscal_year("Analyze the company for 2024"), 2024)

    def test_a_future_year_is_a_forecast_horizon_not_a_filing(self):
        self.assertIsNone(extract_fiscal_year("forecast revenue out to 2099"))
        self.assertIsNone(extract_fiscal_year("terminal growth 2.5%"))


class TestNamesEntity(unittest.TestCase):
    def test_ticker_or_first_distinctive_name_word(self):
        self.assertTrue(names_entity("run a dcf on aapl", "AAPL", "Apple Inc."))
        self.assertTrue(names_entity("Value Apple now", "AAPL", "Apple Inc."))
        self.assertFalse(names_entity("now run the DCF", "AAPL", "Apple Inc."))
        self.assertTrue(names_entity("value amazon", "AMZN", "AMAZON.COM, INC."))


class TestResolveRequestedYear(unittest.TestCase):
    def test_explicit_year_beats_the_extraction(self):
        self.assertEqual(resolve_requested_year(2023, 2025, "dcf", "AAPL", SESSION), RequestedYear(2023, None))

    def test_follow_up_about_the_active_company_keeps_the_session_year(self):
        requested = resolve_requested_year(None, None, "Now run the DCF", "aapl", SESSION)
        self.assertEqual(requested.year, 2024)
        self.assertIn("FY2024", requested.note)

    def test_naming_the_company_again_drops_the_session_year(self):
        self.assertEqual(resolve_requested_year(None, None, "Run a DCF on Apple", "AAPL", SESSION).year, None)

    def test_a_different_company_drops_the_session_year(self):
        self.assertEqual(resolve_requested_year(None, None, "Now run the DCF", "TSLA", SESSION).year, None)

    def test_a_year_in_the_query_drops_the_session_year(self):
        self.assertEqual(resolve_requested_year(None, None, "Now the FY2025 DCF", "AAPL", SESSION).year, None)

    def test_year_zero_is_not_missing(self):
        # The orchestrator's fiscal_year=0 sentinel (fixed in a later change) blocks the follow-up rule today.
        self.assertEqual(resolve_requested_year(0, None, "Now run the DCF", "AAPL", SESSION), RequestedYear(0, None))


class TestChooseFilingYear(unittest.TestCase):
    def test_ingested_year_is_used_as_requested(self):
        self.assertEqual(choose_filing_year("AAPL", 2024, "", AVAILABLE), FilingChoice(2024, 2024, False))

    def test_missing_year_falls_back_to_the_latest_and_is_flagged(self):
        self.assertEqual(choose_filing_year("AAPL", 2020, "", AVAILABLE), FilingChoice(2020, 2025, True))

    def test_no_request_reads_the_year_from_the_query(self):
        self.assertEqual(choose_filing_year("AAPL", 0, "audit fiscal 2024", AVAILABLE), FilingChoice(2024, 2024, False))

    def test_no_request_anywhere_means_the_latest_filing(self):
        self.assertEqual(choose_filing_year("AAPL", None, "audit it", AVAILABLE), FilingChoice(None, 2025, False))


class TestExplainMissingFiling(unittest.TestCase):
    def test_reply_carries_the_reason(self):
        reply = explain_missing_filing("No ingested 10-K filings found for ticker 'ZZZZ'.")
        self.assertTrue(reply.startswith("I am your Senior Equity Research Director. No ingested"))
        self.assertTrue(reply.endswith("upload the requested 10-K filing."))


if __name__ == "__main__":
    unittest.main()
