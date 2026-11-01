"""Tests for the audited history table shown in the forecast task."""

import unittest

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import (
    HistoricalRow,
    build_forecast_baseline,
    render_history_table,
)
from tests.forecast_fixtures import make_aapl_audit


class TestRenderHistoryTable(unittest.TestCase):
    def test_each_audited_year_shows_revenue_growth_and_margin(self):
        b = build_forecast_baseline(make_aapl_audit())
        self.assertIn("| FY2025 | 416,161 | +6.43% | 31.97% |", render_history_table(b.historical_rows))

    def test_a_year_without_growth_shows_a_dash(self):
        table = render_history_table([HistoricalRow(fiscal_year=2023, revenue=100.0, operating_margin_pct=20.0)])
        self.assertIn("| FY2023 | 100 | — | 20.00% |", table)


if __name__ == "__main__":
    unittest.main()
