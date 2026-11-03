"""session_continuity: research turns start a fresh record, chat turns keep theirs; overrides follow both."""

import unittest

from app.agents.specialized.supervisor import SESSION_KEY
from app.agents.specialized.supervisor.modules.session_continuity import SessionFiling, remember_turn

SESSION = {
    "active_ticker": "AAPL", "active_company": "Apple Inc.", "active_fiscal_year": 2024,
    "available_fiscal_years": [2025, 2024], "last_query_type": "full_10k_report", "note": "kept on chat turns",
    SESSION_KEY: {"ticker": "AAPL", "wacc": 0.08},
}
TSLA = SessionFiling(ticker="TSLA", company_name="Tesla, Inc.", fiscal_year=2025, available_fiscal_years=[2025])


class TestRememberTurn(unittest.TestCase):
    def test_research_turn_remembers_only_its_filing_and_report_type(self):
        remembered = remember_turn(SESSION, {}, TSLA, "dcf_valuation_only")
        self.assertEqual(list(remembered), [
            "active_ticker", "active_company", "active_fiscal_year", "available_fiscal_years", "last_query_type",
        ])
        self.assertEqual((remembered["active_ticker"], remembered["last_query_type"]), ("TSLA", "dcf_valuation_only"))

    def test_research_turn_remembers_the_applied_overrides(self):
        overrides = {"ticker": "TSLA", "wacc": 0.09}
        self.assertEqual(remember_turn(SESSION, overrides, TSLA, "dcf_valuation_only")[SESSION_KEY], overrides)

    def test_chat_turn_keeps_the_record_and_updates_the_overrides(self):
        remembered = remember_turn(SESSION, {"ticker": "AAPL", "beta": 1.4})
        self.assertEqual(remembered["note"], "kept on chat turns")
        self.assertEqual(remembered[SESSION_KEY], {"ticker": "AAPL", "beta": 1.4})

    def test_chat_turn_with_no_overrides_left_forgets_them(self):
        self.assertNotIn(SESSION_KEY, remember_turn(SESSION, {}))
        self.assertIn(SESSION_KEY, SESSION)  # the input record is never changed


if __name__ == "__main__":
    unittest.main()
