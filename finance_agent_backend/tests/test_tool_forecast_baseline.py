"""Tests for the forecast baseline built from the Financial Auditor output."""

import unittest

from app.agents.tools.financial_math_tools import NormalizationAdjustment
from app.agents.tools.forecast_baseline import build_forecast_baseline, render_history_table
from tests.forecast_fixtures import (
    AAPL_BALANCE_SHEET,
    AAPL_ROWS,
    make_aapl_audit,
    make_audit,
    make_loss_audit,
    make_low_margin_audit,
)


class TestForecastBaseline(unittest.TestCase):
    def test_converts_auditor_percentages_once(self):
        b = build_forecast_baseline(make_aapl_audit())
        self.assertEqual(b.base_year, 2025)
        self.assertEqual(b.base_revenue, 416161.0)
        self.assertAlmostEqual(b.base_operating_margin, 0.3197, places=4)
        self.assertEqual(b.margin_source, "reported")
        self.assertAlmostEqual(b.latest_revenue_growth, 0.0643, places=4)
        self.assertAlmostEqual(b.historical_cagr, 0.0420, places=4)
        self.assertAlmostEqual(b.growth_volatility, 0.02205, places=4)
        self.assertEqual(b.history_years, 3)

    def test_low_margin_and_low_tax_stay_small(self):
        b = build_forecast_baseline(make_low_margin_audit())
        self.assertAlmostEqual(b.base_operating_margin, 0.007, places=5)
        self.assertAlmostEqual(b.tax_rate, 0.008, places=5)
        self.assertEqual(b.tax_source, "median_effective")

    def test_tax_rate_is_median_of_profitable_years(self):
        # FY2023 14.72%, FY2024 24.09% (one-off), FY2025 15.60% -> median 15.60%
        b = build_forecast_baseline(make_aapl_audit())
        self.assertAlmostEqual(b.tax_rate, 0.156, places=4)
        self.assertEqual(b.tax_source, "median_effective")

    def test_loss_years_fall_back_to_statutory_tax(self):
        b = build_forecast_baseline(make_loss_audit())
        self.assertEqual(b.tax_source, "statutory_default")
        self.assertAlmostEqual(b.tax_rate, 0.21)
        self.assertAlmostEqual(b.base_operating_margin, -0.05, places=4)

    def test_normalized_margin_used_when_adjusted(self):
        adj = NormalizationAdjustment(
            fiscal_year=2025, label="Litigation settlement charge", amount=4161.61,
            direction="depressed_reported_earnings", affects="operating_income",
        )
        b = build_forecast_baseline(make_aapl_audit(normalization=[adj]))
        self.assertEqual(b.margin_source, "normalized")
        self.assertAlmostEqual(b.base_operating_margin, 0.3297, places=4)

    def test_capex_is_multi_year_average(self):
        b = build_forecast_baseline(make_aapl_audit())
        self.assertAlmostEqual(b.capex_pct_of_revenue, 0.02777, places=4)
        self.assertEqual(b.capex_years, 3)

    def test_da_reported_or_capex_parity_proxy(self):
        reported = build_forecast_baseline(make_aapl_audit())
        self.assertEqual(reported.da_source, "reported")
        self.assertAlmostEqual(reported.da_pct_of_revenue, 0.0275, places=4)

        proxy = build_forecast_baseline(make_aapl_audit(include_depreciation=False))
        self.assertEqual(proxy.da_source, "capex_parity_proxy")
        self.assertAlmostEqual(proxy.da_pct_of_revenue, proxy.capex_pct_of_revenue)
        self.assertTrue(any("D&A not reported" in w for w in proxy.warnings))

    def test_working_capital_intensity_from_days(self):
        # (36.5 + (10.0 - 109.5) * (1 - 0.4691)) / 365 = -0.0447
        b = build_forecast_baseline(make_aapl_audit(working_capital=True))
        self.assertEqual(b.nwc_source, "working_capital_days")
        self.assertAlmostEqual(b.nwc_pct_of_incremental_revenue, -0.0447, places=4)

        none = build_forecast_baseline(make_aapl_audit())
        self.assertEqual(none.nwc_source, "not_available")
        self.assertEqual(none.nwc_pct_of_incremental_revenue, 0.0)

    def test_partial_working_capital_days_not_used(self):
        # Receivables extracted but payables missing: a DSO-only intensity would flip the sign.
        rows = [dict(r) for r in AAPL_ROWS]
        rows[-1].update(accounts_receivable=41616.1, inventories=6053.7)
        b = build_forecast_baseline(make_audit("AAPL", rows, AAPL_BALANCE_SHEET))
        self.assertEqual(b.nwc_source, "not_available")
        self.assertEqual(b.nwc_pct_of_incremental_revenue, 0.0)
        self.assertTrue(any("working capital" in w.lower() for w in b.warnings))

    def test_sbc_intensity(self):
        b = build_forecast_baseline(make_aapl_audit(sbc=12863.0))
        self.assertEqual(b.sbc_source, "reported")
        self.assertAlmostEqual(b.sbc_pct_of_revenue, 12863.0 / 416161.0, places=6)
        self.assertEqual(build_forecast_baseline(make_aapl_audit()).sbc_source, "not_reported")

    def test_single_audited_year(self):
        b = build_forecast_baseline(make_aapl_audit(years=(2025,)))
        self.assertEqual(b.history_years, 1)
        self.assertIsNone(b.historical_cagr)
        self.assertIsNone(b.latest_revenue_growth)
        self.assertIsNone(b.growth_volatility)
        self.assertEqual(b.capex_years, 1)

    def test_history_rows_and_table(self):
        b = build_forecast_baseline(make_aapl_audit())
        self.assertEqual([r.fiscal_year for r in b.historical_rows], [2023, 2024, 2025])
        self.assertIn("| FY2025 | 416,161 | +6.43% | 31.97% |", render_history_table(b.historical_rows))

    def test_invested_capital_is_debt_plus_equity(self):
        self.assertAlmostEqual(build_forecast_baseline(make_aapl_audit()).base_invested_capital, 152393.0)


if __name__ == "__main__":
    unittest.main()
