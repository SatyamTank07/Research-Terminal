"""Tests for the forecast baseline built from the Financial Auditor output."""

import unittest

from app.agents.specialized.financial_auditor import NormalizationAdjustment
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.specialized.forecasting_analyst.modules.audited_baseline.thresholds import (
    STEADY_STATE_RONIC_CAP,
    STEADY_STATE_RONIC_DEFAULT,
    STEADY_STATE_RONIC_FLOOR,
)
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

    def test_capex_start_is_recency_weighted(self):
        # FY2023 2.859%, FY2024 2.416%, FY2025 3.055% -> weights 1:2:3 = 2.809%; simple mean 2.777%
        b = build_forecast_baseline(make_aapl_audit())
        self.assertAlmostEqual(b.capex_pct_of_revenue, 0.028095, places=5)
        self.assertAlmostEqual(b.capex_average_pct_of_revenue, 0.02777, places=4)
        self.assertAlmostEqual(b.latest_capex_pct_of_revenue, 12715.0 / 416161.0, places=6)
        self.assertEqual(b.capex_years, 3)

    def test_two_year_capex_uses_the_latest_weights(self):
        # Fewer audited years take the tail of the 1:2:3 weights: 2:3, latest 60%.
        b = build_forecast_baseline(make_aapl_audit(years=(2024, 2025)))
        expected = (2 * 9447.0 / 391035.0 + 3 * 12715.0 / 416161.0) / 5
        self.assertAlmostEqual(b.capex_pct_of_revenue, expected, places=6)

    def test_da_reported_or_capex_parity_proxy(self):
        reported = build_forecast_baseline(make_aapl_audit())
        self.assertEqual(reported.da_source, "reported")
        self.assertAlmostEqual(reported.da_pct_of_revenue, 0.0275, places=4)

        proxy = build_forecast_baseline(make_aapl_audit(include_depreciation=False))
        self.assertEqual(proxy.da_source, "capex_parity_proxy")
        self.assertAlmostEqual(proxy.da_pct_of_revenue, proxy.capex_average_pct_of_revenue)
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

    def test_history_rows_cover_every_audited_year(self):
        b = build_forecast_baseline(make_aapl_audit())
        self.assertEqual([r.fiscal_year for r in b.historical_rows], [2023, 2024, 2025])

    def test_invested_capital_is_debt_plus_equity(self):
        self.assertAlmostEqual(build_forecast_baseline(make_aapl_audit()).base_invested_capital, 152393.0)

    def test_steady_state_ronic_is_base_roic_clamped(self):
        aapl = build_forecast_baseline(make_aapl_audit())
        self.assertGreater(aapl.base_roic, STEADY_STATE_RONIC_CAP)  # ~74%: asset-light
        self.assertEqual((aapl.steady_state_ronic, aapl.steady_state_ronic_source), (0.30, "base_roic_capped"))

        low = build_forecast_baseline(make_low_margin_audit())  # 0.7% margin on 9,500 capital -> ~8.0%
        self.assertGreater(low.base_roic, 0.0)
        self.assertEqual(low.steady_state_ronic, max(low.base_roic, STEADY_STATE_RONIC_FLOOR))

    def test_loss_maker_gets_default_ronic_and_warning(self):
        loss = build_forecast_baseline(make_loss_audit())
        self.assertLess(loss.base_roic, 0.0)
        self.assertEqual(loss.steady_state_ronic, STEADY_STATE_RONIC_DEFAULT)
        self.assertEqual(loss.steady_state_ronic_source, "default_no_positive_roic")
        self.assertTrue(any("ROIC is not positive" in w for w in loss.warnings))

    def test_negative_invested_capital_has_no_roic(self):
        balance = {**AAPL_BALANCE_SHEET, "stockholders_equity": -150000.0}
        b = build_forecast_baseline(make_audit("NEG", [dict(r) for r in AAPL_ROWS], balance))
        self.assertIsNone(b.base_roic)
        self.assertIsNone(b.base_invested_capital)
        self.assertEqual(b.steady_state_ronic_source, "default_no_positive_roic")

    def test_margin_trend_classification(self):
        self.assertEqual(build_forecast_baseline(make_aapl_audit()).margin_trend, "expanding")
        self.assertEqual(build_forecast_baseline(make_loss_audit()).margin_trend, "expanding")  # -10% -> -5%
        self.assertEqual(build_forecast_baseline(make_aapl_audit(years=(2025,))).margin_trend, "insufficient_history")
        self.assertIsNone(build_forecast_baseline(make_aapl_audit(years=(2025,))).historical_average_margin)

        rows = [dict(r) for r in AAPL_ROWS]
        rows[1]["operating_income"] = 150000.0  # spike to 38.4% then give-back to 32.0%
        mixed = build_forecast_baseline(make_audit("MIX", rows, AAPL_BALANCE_SHEET))
        self.assertEqual(mixed.margin_trend, "mixed")
        self.assertAlmostEqual(mixed.historical_average_margin, (0.2982 + 0.3836 + 0.3197) / 3, places=3)

        falling = [dict(r) for r in AAPL_ROWS]
        falling[0]["operating_income"], falling[2]["operating_income"] = 133050.0, 114301.0
        self.assertEqual(build_forecast_baseline(make_audit("DN", falling, AAPL_BALANCE_SHEET)).margin_trend, "contracting")


if __name__ == "__main__":
    unittest.main()
