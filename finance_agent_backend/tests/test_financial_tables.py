"""Unit tests for the Item 8 statement-table parser (`app/services/financial_tables.py`).

Validates legacy `Col_n` markdown (split `$` cells, colspan drift), date headers, clean rendering
round trips and raw ingestion grid rows.
"""

import unittest

from app.services.financial_tables import (
    is_well_formed,
    parse_statement_rows,
    parse_statement_table,
    parse_value_token,
    render_clean_statement,
)
from tests.statement_fixtures import TSLA_FY2025_BALANCE_SHEET, TSLA_FY2025_INCOME_STATEMENT


class TestFinancialTableParsing(unittest.TestCase):
    def test_01_value_tokens(self):
        self.assertEqual(parse_value_token("( 5,001 )"), -5001.0)
        self.assertEqual(parse_value_token("—"), 0.0)
        self.assertIsNone(parse_value_token("$"))
        self.assertIsNone(parse_value_token(""))
        self.assertEqual(parse_value_token("$3,794"), 3794.0)
        self.assertEqual(parse_value_token("1.08"), 1.08)

    def test_02_legacy_income_statement_periods_and_alignment(self):
        grid = parse_statement_table(TSLA_FY2025_INCOME_STATEMENT)
        self.assertEqual(grid.periods, [2025, 2024, 2023])
        self.assertTrue(is_well_formed(grid))
        rows = {}
        for r in grid.value_rows:
            rows.setdefault(r.norm_label, r)  # first occurrence (revenue section, not cost section)
        self.assertEqual(rows["automotive sales"].values, {2025: 65821.0, 2024: 72480.0, 2023: 78509.0})
        self.assertEqual(rows["automotive sales"].section, "revenues")
        self.assertEqual(rows["automotive regulatory credits"].values[2024], 2763.0)
        self.assertEqual(rows["interest expense"].values[2025], -338.0)
        self.assertEqual(rows["restructuring and other"].values[2023], 0.0)

    def test_03_balance_sheet_date_headers(self):
        grid = parse_statement_table(TSLA_FY2025_BALANCE_SHEET)
        self.assertEqual(grid.periods, [2025, 2024])

    def test_04_clean_rendering_round_trip(self):
        grid = parse_statement_table(TSLA_FY2025_INCOME_STATEMENT)
        md = render_clean_statement(grid)
        self.assertIn("| Line item | FY2025 | FY2024 | FY2023 |", md)
        self.assertIn("| Total revenues | 94,827 | 97,690 | 96,773 |", md)
        self.assertIn("| Provision for (benefit from) income taxes | 1,423 | 1,837 | (5,001) |", md)
        self.assertNotIn("Col_", md)
        reparsed = parse_statement_table(md)
        self.assertEqual(reparsed.periods, [2025, 2024, 2023])
        self.assertEqual(
            {r.norm_label: r.values for r in reparsed.value_rows}["total revenues"],
            {2025: 94827.0, 2024: 97690.0, 2023: 96773.0},
        )

    def test_05_raw_grid_rows_like_ingestion(self):
        rows = [
            ["", "2025", "", "2024"],
            ["Total revenues", "$", "100", "$ 90"],
            ["Net income", "(5", ")", "7"],
        ]
        grid = parse_statement_rows(rows)
        self.assertEqual(grid.periods, [2025, 2024])
        vals = {r.norm_label: r.values for r in grid.value_rows}
        self.assertEqual(vals["total revenues"], {2025: 100.0, 2024: 90.0})
        self.assertEqual(vals["net income"], {2025: -5.0, 2024: 7.0})


if __name__ == "__main__":
    unittest.main()
