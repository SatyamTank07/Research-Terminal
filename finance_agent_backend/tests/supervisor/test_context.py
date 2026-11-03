"""The Supervisor's catalog reads against the real database: available years, the chosen filing, the error hint."""

import unittest

from app.agents.specialized.supervisor.context import (
    FilingCatalogError,
    active_session,
    load_filing,
    load_filing_catalog,
)


class TestActiveSession(unittest.TestCase):
    def test_pending_action_is_never_carried(self):
        self.assertEqual(active_session({"active_ticker": "AAPL", "pending_action": "x"}), {"active_ticker": "AAPL"})
        self.assertEqual(active_session(None), {})


class TestFilingCatalog(unittest.TestCase):
    def test_available_years_are_newest_first(self):
        years = load_filing_catalog(" aapl ")
        self.assertIn(2025, years)
        self.assertEqual(years, sorted(years, reverse=True))

    def test_missing_ticker_raises_with_the_catalog_hint(self):
        with self.assertRaises(FilingCatalogError) as ctx:
            load_filing_catalog(None)
        self.assertIn("Could not identify a target stock ticker from the request", str(ctx.exception))
        self.assertIn("Ingested 10-K tickers include:", str(ctx.exception))

    def test_unknown_ticker_raises_a_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            load_filing_catalog("UNKNOWN_CORP_XYZ")
        self.assertIn("No ingested 10-K filings found for ticker 'UNKNOWN_CORP_XYZ'", str(ctx.exception))


class TestLoadFiling(unittest.TestCase):
    def test_requested_year_is_loaded(self):
        filing = load_filing("AAPL", 2025)
        self.assertEqual((filing.ticker, filing.fiscal_year), ("AAPL", 2025))
        self.assertIn("Apple", filing.company_name)
        self.assertTrue(filing.document_id)

    def test_year_without_a_filing_falls_back_to_the_latest(self):
        self.assertEqual(load_filing("AAPL", 1990).fiscal_year, load_filing_catalog("AAPL")[0])


if __name__ == "__main__":
    unittest.main()
