"""Unit and integration tests for rag_table_tools.py (Milestone 1, Tool 1).

Validates deterministic SQL table retrieval across real ingested 10-K filings:
- Apple Inc. (AAPL FY2025 & FY2024)
- Tesla, Inc. (TSLA FY2025)
"""

import unittest
from app.agents.tools.rag_table_tools import (
    TableChunkResult,
    retrieve_10k_tables,
    retrieve_multiyear_financial_series,
    retrieve_10k_tables_tool,
)


class TestRAGTableTools(unittest.TestCase):
    """Test suite verifying universal GAAP 10-K table retrieval."""

    def test_01_aapl_2025_income_statement(self):
        """Verify AAPL FY2025 Income Statement retrieval and multi-year columns."""
        results = retrieve_10k_tables(
            ticker="AAPL",
            fiscal_year=2025,
            statement_type="income_statement",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return at least one table for AAPL income statement")
        
        primary = results[0]
        self.assertIsInstance(primary, TableChunkResult)
        self.assertEqual(primary.ticker, "AAPL")
        self.assertEqual(primary.fiscal_year, 2025)
        self.assertIn("Item 8", primary.item)
        self.assertTrue(len(primary.chunk_id) > 0)
        self.assertTrue(len(primary.breadcrumb) > 0)
        self.assertGreater(primary.row_count, 0)
        self.assertGreater(primary.column_count, 0)

        # Content verification: GAAP rows and multi-year dates
        content_lower = primary.table_markdown.lower()
        self.assertIn("net sales", content_lower)
        self.assertIn("operating income", content_lower)
        self.assertIn("2025", primary.table_markdown)
        self.assertIn("2024", primary.table_markdown)
        self.assertIn("2023", primary.table_markdown)

    def test_02_aapl_2025_balance_sheet(self):
        """Verify AAPL FY2025 Consolidated Balance Sheet retrieval."""
        results = retrieve_10k_tables(
            ticker="AAPL",
            fiscal_year=2025,
            statement_type="balance_sheet",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return at least one table for AAPL balance sheet")
        
        primary = results[0]
        content_lower = primary.table_markdown.lower()
        self.assertIn("current assets", content_lower)
        self.assertIn("cash and cash equivalents", content_lower)
        self.assertIn("total assets", content_lower)

    def test_03_aapl_2025_cash_flows(self):
        """Verify AAPL FY2025 Statement of Cash Flows retrieval."""
        results = retrieve_10k_tables(
            ticker="AAPL",
            fiscal_year=2025,
            statement_type="cash_flow",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return at least one table for AAPL cash flows")
        
        primary = results[0]
        content_lower = primary.table_markdown.lower()
        self.assertIn("operating activities", content_lower)
        self.assertIn("investing activities", content_lower)

    def test_04_tsla_2025_income_statement(self):
        """Verify TSLA FY2025 Income Statement retrieval without company hardcoding."""
        results = retrieve_10k_tables(
            ticker="TSLA",
            fiscal_year=2025,
            statement_type="income_statement",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return at least one table for TSLA income statement")
        
        primary = results[0]
        self.assertEqual(primary.ticker, "TSLA")
        self.assertEqual(primary.fiscal_year, 2025)
        content_lower = primary.table_markdown.lower()
        self.assertIn("total revenues", content_lower)
        self.assertIn("net income", content_lower)
        self.assertIn("2025", primary.table_markdown)
        self.assertIn("2024", primary.table_markdown)
        self.assertIn("2023", primary.table_markdown)

    def test_05_tsla_2025_balance_sheet(self):
        """Verify TSLA FY2025 Balance Sheet retrieval."""
        results = retrieve_10k_tables(
            ticker="TSLA",
            fiscal_year=2025,
            statement_type="balance_sheet",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return at least one table for TSLA balance sheet")
        
        primary = results[0]
        content_lower = primary.table_markdown.lower()
        self.assertIn("current assets", content_lower)
        self.assertIn("cash and cash equivalents", content_lower)
        self.assertIn("total assets", content_lower)

    def test_06_tsla_2025_cash_flows(self):
        """Verify TSLA FY2025 Statement of Cash Flows retrieval."""
        results = retrieve_10k_tables(
            ticker="TSLA",
            fiscal_year=2025,
            statement_type="cash_flow",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return at least one table for TSLA cash flows")
        
        primary = results[0]
        content_lower = primary.table_markdown.lower()
        self.assertIn("cash flows from operating activities", content_lower)

    def test_07_cross_filing_time_series_stitching(self):
        """Verify cross-filing retrieval across AAPL FY2025 and FY2024 (Latest Precedence)."""
        results = retrieve_multiyear_financial_series(
            ticker="AAPL",
            statement_type="income_statement",
            section_item="Item 8",
            limit=6,
        )
        self.assertGreaterEqual(len(results), 2, "Should return tables from both FY2025 and FY2024 filings")
        
        years_found = [r.fiscal_year for r in results]
        self.assertIn(2025, years_found)
        self.assertIn(2024, years_found)
        
        # Verify newest filing appears first
        self.assertEqual(results[0].fiscal_year, 2025)

    def test_08_dynamic_keyword_search(self):
        """Verify dynamic caller keyword search (e.g. searching for Debt or Leases)."""
        results = retrieve_10k_tables(
            ticker="AAPL",
            fiscal_year=2025,
            keyword_filter="Total shareholders’ equity",
            section_item="Item 8",
            limit=2,
        )
        self.assertGreater(len(results), 0, "Should find equity schedule via keyword filter")
        self.assertIn("shareholders’ equity", results[0].table_markdown.lower())

    def test_09_langchain_tool_wrapper(self):
        """Verify LangChain @tool wrapper executes cleanly and returns serializable dicts."""
        tool_output = retrieve_10k_tables_tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "statement_type": "income_statement",
            "section_item": "Item 8",
        })
        self.assertIsInstance(tool_output, list)
        self.assertGreater(len(tool_output), 0)
        self.assertIn("chunk_id", tool_output[0])
        self.assertIn("table_markdown", tool_output[0])
        self.assertIn("breadcrumb", tool_output[0])


    def test_10_nvda_2026_income_statement_fallback(self):
        """Verify NVDA FY2026 Income Statement retrieval via automatic section fallback from Item 8."""
        results = retrieve_10k_tables(
            ticker="NVDA",
            fiscal_year=2026,
            statement_type="income_statement",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return tables for NVDA income statement via fallback")
        primary = results[0]
        self.assertEqual(primary.ticker, "NVDA")
        self.assertEqual(primary.fiscal_year, 2026)
        content_lower = primary.table_markdown.lower()
        self.assertTrue("net income" in content_lower or "revenue" in content_lower)

    def test_11_nvda_2026_balance_sheet_fallback(self):
        """Verify NVDA FY2026 Balance Sheet retrieval via automatic section fallback."""
        results = retrieve_10k_tables(
            ticker="NVDA",
            fiscal_year=2026,
            statement_type="balance_sheet",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return tables for NVDA balance sheet via fallback")
        primary = results[0]
        content_lower = primary.table_markdown.lower()
        self.assertIn("cash and cash equivalents", content_lower)
        self.assertTrue("marketable securities" in content_lower or "total assets" in content_lower)

    def test_12_nvda_2026_cash_flows_fallback(self):
        """Verify NVDA FY2026 Cash Flows retrieval via automatic section fallback."""
        results = retrieve_10k_tables(
            ticker="NVDA",
            fiscal_year=2026,
            statement_type="cash_flow",
            section_item="Item 8",
            limit=3,
        )
        self.assertGreater(len(results), 0, "Should return tables for NVDA cash flows via fallback")
        primary = results[0]
        content_lower = primary.table_markdown.lower()
        self.assertIn("operating activities", content_lower)


if __name__ == "__main__":
    unittest.main()
