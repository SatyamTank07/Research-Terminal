"""Unit tests for the deterministic forecast schedule engine."""

import unittest

from app.agents.tools.dcf_tools import calculate_dcf_with_sensitivity
from app.agents.tools.forecast_tools import HistoricalRow, calculate_forecast_schedule

BASE = dict(base_revenue=1000.0, base_year=2025, tax_rate=0.25, capex_pct_of_revenue=0.05, depreciation_pct_of_revenue=0.04)


class TestForecastSchedule(unittest.TestCase):
    def test_compounding_and_ufcf_bridge(self):
        res = calculate_forecast_schedule(revenue_growth_rates=[0.10, 0.10], operating_margins=[0.20, 0.20], **BASE)
        y1, y2 = res["forecast_schedule"]
        self.assertAlmostEqual(y1["projected_revenue"], 1100.0)
        self.assertAlmostEqual(y2["projected_revenue"], 1210.0)
        self.assertAlmostEqual(y1["projected_nopat"], 165.0)
        self.assertAlmostEqual(y1["projected_unlevered_fcf"], 154.0)
        self.assertAlmostEqual(y2["projected_unlevered_fcf"], 169.4)
        self.assertAlmostEqual(res["cumulative_fcf"], 323.4)
        self.assertAlmostEqual(res["average_annual_fcf"], 161.7)
        self.assertAlmostEqual(res["revenue_cagr_pct"], 10.0)
        self.assertEqual(res["forecast_horizon_years"], 2)
        self.assertNotIn("provenance_mode", res)

    def test_working_capital_reduces_ufcf_and_sbc_is_memo(self):
        res = calculate_forecast_schedule(
            revenue_growth_rates=[0.10], operating_margins=[0.20],
            nwc_change_pct_of_revenue=0.10, sbc_pct_of_revenue=0.02, **BASE,
        )
        y1 = res["forecast_schedule"][0]
        self.assertAlmostEqual(y1["projected_nwc_change"], 10.0)
        self.assertAlmostEqual(y1["projected_sbc"], 22.0)
        # GAAP EBIT already expenses SBC: 165 NOPAT + 44 D&A - 55 CapEx - 10 dNWC
        self.assertAlmostEqual(y1["projected_unlevered_fcf"], 144.0)
        self.assertIn("| Stock-Based Compensation (memo, in EBIT) |", res["forecast_table_markdown"])

    def test_per_year_intensity_lists(self):
        res = calculate_forecast_schedule(
            revenue_growth_rates=[0.0, 0.0], operating_margins=[0.2, 0.2],
            **{**BASE, "capex_pct_of_revenue": [0.05, 0.10]},
        )
        self.assertEqual([y["projected_capex"] for y in res["forecast_schedule"]], [50.0, 100.0])

    def test_rejects_whole_percent_inputs(self):
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(revenue_growth_rates=[7.0], operating_margins=[0.2], **BASE)
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(revenue_growth_rates=[0.07], operating_margins=[32.0], **BASE)
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(revenue_growth_rates=[0.07], operating_margins=[0.2], **{**BASE, "tax_rate": 15.6})

    def test_depreciation_is_required(self):
        args = {k: v for k, v in BASE.items() if k != "depreciation_pct_of_revenue"}
        with self.assertRaises(TypeError):
            calculate_forecast_schedule(revenue_growth_rates=[0.05], operating_margins=[0.2], **args)

    def test_rejects_bad_shapes_and_values(self):
        cases = [
            {"revenue_growth_rates": [], "operating_margins": []},
            {"revenue_growth_rates": [0.05] * 11, "operating_margins": [0.2] * 11},
            {"revenue_growth_rates": [0.05, 0.05], "operating_margins": [0.2]},
            {"revenue_growth_rates": [float("nan")], "operating_margins": [0.2]},
            {"revenue_growth_rates": [-0.96], "operating_margins": [0.2]},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                calculate_forecast_schedule(**case, **BASE)

    def test_base_revenue_and_year_validation(self):
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(revenue_growth_rates=[0.05], operating_margins=[0.2], **{**BASE, "base_revenue": 0.0})
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(revenue_growth_rates=[0.05], operating_margins=[0.2], **{**BASE, "base_year": 25})

    def test_ten_year_horizon(self):
        res = calculate_forecast_schedule(revenue_growth_rates=[0.05] * 10, operating_margins=[0.2] * 10, **BASE)
        self.assertEqual(res["forecast_horizon_years"], 10)
        self.assertEqual(res["forecast_schedule"][-1]["projected_year"], 2035)

    def test_table_history_columns_and_optional_rows(self):
        rows = [
            HistoricalRow(fiscal_year=2024, revenue=900.0, revenue_growth_pct=None, operating_margin_pct=19.0),
            HistoricalRow(fiscal_year=2025, revenue=1000.0, revenue_growth_pct=11.11, operating_margin_pct=20.0),
        ]
        res = calculate_forecast_schedule(
            revenue_growth_rates=[0.10, 0.08], operating_margins=[0.20, 0.20], historical_rows=rows, **BASE,
        )
        table = res["forecast_table_markdown"]
        self.assertIn("| Metric ($ Millions) | FY2024A | FY2025A | FY2026E | FY2027E |", table)
        self.assertIn("| Revenue | $900 | $1,000 | $1,100 | $1,188 |", table)
        self.assertIn("| Depreciation & Amortization |", table)
        self.assertNotIn("Stock-Based Compensation", table)
        self.assertNotIn("ΔNWC", table)

    def test_handoff_into_dcf(self):
        res = calculate_forecast_schedule(
            revenue_growth_rates=[0.06, 0.05, 0.04, 0.03, 0.025], operating_margins=[0.2] * 5, **BASE,
        )
        dcf = calculate_dcf_with_sensitivity(
            projected_fcfs=res["projected_fcfs"], wacc=0.09, terminal_growth_rate=0.025,
            net_debt=0.0, diluted_shares=100.0, mid_year_convention=True,
        )
        self.assertGreater(dcf["fair_value_per_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
