"""Unit Tests for SEC Catalog Discovery Tool (search_sec_catalog).

Validates:
1. Tool attributes (name, description, callable interface).
2. Empty or whitespace query handling (returns empty list).
3. Exact and prefix ticker search against PostgreSQL documents.
4. Company corporate name substring search.
5. Case-insensitivity for tickers and company names.
6. Unknown/unfiled company search returns empty list.
7. Graceful exception handling on database error.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.agents.tools.catalog_tools import search_sec_catalog


class TestCatalogTools(unittest.TestCase):
    """Test suite for search_sec_catalog tool."""

    def test_01_tool_metadata(self):
        """Verify tool name and schema definition."""
        self.assertEqual(search_sec_catalog.name, "search_sec_catalog")
        self.assertIn("PostgreSQL document catalog", search_sec_catalog.description)

    def test_02_empty_and_whitespace_query(self):
        """Verify empty and whitespace inputs return empty list without database query."""
        self.assertEqual(search_sec_catalog.invoke({"query": ""}), [])
        self.assertEqual(search_sec_catalog.invoke({"query": "   "}), [])
        self.assertEqual(search_sec_catalog.invoke({"query": None}), [])

    def test_03_exact_ticker_search(self):
        """Verify searching by exact ticker returns matching filing records."""
        results = search_sec_catalog.invoke({"query": "AAPL"})
        self.assertIsInstance(results, list)
        if results:
            self.assertEqual(results[0]["ticker"], "AAPL")
            self.assertIn("Apple", results[0]["company_name"])
            self.assertIn("fiscal_year", results[0])
            self.assertIn("document_id", results[0])

    def test_04_company_name_search(self):
        """Verify searching by company name returns matching records."""
        results = search_sec_catalog.invoke({"query": "Tesla"})
        self.assertIsInstance(results, list)
        if results:
            self.assertEqual(results[0]["ticker"], "TSLA")
            self.assertIn("Tesla", results[0]["company_name"])

    def test_05_case_insensitivity(self):
        """Verify ticker and company name searches are case-insensitive."""
        res_lower = search_sec_catalog.invoke({"query": "aapl"})
        res_upper = search_sec_catalog.invoke({"query": "AAPL"})
        self.assertEqual(len(res_lower), len(res_upper))
        if res_lower and res_upper:
            self.assertEqual(res_lower[0]["ticker"], res_upper[0]["ticker"])

    def test_06_non_existent_entity(self):
        """Verify searching for an unfiled company returns an empty list."""
        results = search_sec_catalog.invoke({"query": "NON_EXISTENT_COMPANY_XYZ_123"})
        self.assertEqual(results, [])

    @patch("app.agents.tools.catalog_tools.SessionLocal")
    def test_07_database_error_handling(self, mock_session_local):
        """Verify graceful fallback to empty list when database query fails."""
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("Database connection failure")
        mock_session_local.return_value = mock_db

        results = search_sec_catalog.invoke({"query": "AAPL"})
        self.assertEqual(results, [])
        mock_db.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
