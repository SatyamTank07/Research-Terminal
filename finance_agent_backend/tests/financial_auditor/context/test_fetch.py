"""Audit window and statement fetch: which filings the Financial Auditor reads."""

import unittest
from unittest.mock import MagicMock, patch

from app.agents.specialized.financial_auditor.context.fetch import (
    STATEMENT_TYPES,
    fetch_statement_tables,
    resolve_audit_window,
)

RETRIEVE = "app.agents.specialized.financial_auditor.context.fetch.retrieve_10k_tables"


class TestAuditWindow(unittest.TestCase):
    def test_three_years_ending_at_the_target_year_by_default(self):
        self.assertEqual(resolve_audit_window(2025, None), [2023, 2024, 2025])

    def test_five_years_when_every_filing_in_the_span_exists(self):
        self.assertEqual(resolve_audit_window(2025, [2025, 2024, 2023, 2022, 2021]), [2021, 2022, 2023, 2024, 2025])

    def test_four_years_when_the_fifth_filing_is_missing(self):
        self.assertEqual(resolve_audit_window(2025, [2025, 2024, 2023, 2022]), [2022, 2023, 2024, 2025])

    def test_a_gap_in_the_filings_keeps_the_three_year_window(self):
        self.assertEqual(resolve_audit_window(2025, [2025, 2024, 2022]), [2023, 2024, 2025])

    def test_later_filings_do_not_extend_the_window(self):
        self.assertEqual(resolve_audit_window(2023, [2026, 2025, 2024, 2023]), [2021, 2022, 2023])


class TestFetchStatementTables(unittest.TestCase):
    def test_reads_the_target_filing_and_up_to_two_prior_filings_newest_first(self):
        hit = MagicMock()
        hit.model_dump.return_value = {"chunk_id": "c"}
        with patch("app.database.SessionLocal") as session, patch(RETRIEVE, return_value=[hit]) as retrieve:
            tables = fetch_statement_tables("AAPL", 2025, [2025, 2024, 2023, 2022])
        years = [c.args[1] for c in retrieve.call_args_list]
        self.assertEqual(years, [y for y in (2025, 2024, 2023) for _ in STATEMENT_TYPES])
        self.assertEqual({st: len(v) for st, v in tables.items()}, {st: 3 for st in STATEMENT_TYPES})
        session.return_value.close.assert_called_once()

    def test_a_database_error_returns_no_tables(self):
        with patch("app.database.SessionLocal"), patch(RETRIEVE, side_effect=RuntimeError("db down")):
            self.assertEqual(fetch_statement_tables("AAPL", 2025, [2025]), {})


if __name__ == "__main__":
    unittest.main()
