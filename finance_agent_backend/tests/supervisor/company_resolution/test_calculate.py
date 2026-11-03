"""company_resolution: the caller's ticker wins, then the judge's extraction, then the session; none -> ask."""

import unittest

from app.agents.specialized.supervisor.modules.company_resolution import ResolvedCompany, resolve_company


class TestResolveCompany(unittest.TestCase):
    def test_explicit_ticker_beats_the_extraction_and_the_session(self):
        self.assertEqual(resolve_company("TSLA", "AAPL", "NVDA"), ResolvedCompany("TSLA", "explicit"))

    def test_extracted_ticker_beats_the_session(self):
        self.assertEqual(resolve_company("", "AAPL", "NVDA"), ResolvedCompany("AAPL", "extracted"))

    def test_follow_up_without_a_company_continues_with_the_session_ticker(self):
        self.assertEqual(resolve_company(None, None, "NVDA"), ResolvedCompany("NVDA", "session"))

    def test_no_ticker_anywhere_means_ask_which_company(self):
        self.assertIsNone(resolve_company("", None, None))

    def test_a_blank_extraction_is_passed_on_for_the_catalog_to_reject(self):
        self.assertEqual(resolve_company(None, " ", "NVDA"), ResolvedCompany(" ", "extracted"))


if __name__ == "__main__":
    unittest.main()
