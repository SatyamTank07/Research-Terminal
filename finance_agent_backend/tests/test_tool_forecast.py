"""Unit and Integration Tests for forecast_tools.py (Milestone 4, Tool 1).

Validates:
1. Deterministic 5-year financial forecasting across real 10-K benchmarks (AAPL, TSLA, NVDA).
2. Option C Flexible Dual-Mode execution:
   - 'simplified_nopat_less_capex' (provenance tagged)
   - 'comprehensive_line_item' (including D&A and Working Capital changes)
3. Direct pipeline handoff of projected_fcfs into calculate_dcf_with_sensitivity (Milestone 1/3 engine).
4. Publication-ready Markdown comparison table generation.
5. Defensive guards: auto-normalization of whole percentages, NaN/Inf rejection, and horizon bounds.
6. LangChain @tool wrapper invocation and schema fidelity.
"""

import unittest
import math
from app.agents.tools.forecast_tools import (
    ForecastScheduleResult,
    ForecastYearResult,
    calculate_forecast_schedule,
    calculate_forecast_schedule_tool,
)
from app.agents.tools.dcf_tools import calculate_dcf_with_sensitivity


class TestForecastTools(unittest.TestCase):
    """Test suite for deterministic financial forecast schedule engine."""

    def test_01_apple_fy2025_simplified_forecast(self):
        """
        Verify Apple FY2025 5-year forecast in simplified mode:
        Base Revenue: $416,161.0M
        Effective Tax: 15.61%
        CapEx: 3.06%
        Growth: 7.0% -> 4.5%
        Margins: 32.0% -> 33.0%
        """
        base_rev = 416161.0
        base_year = 2025
        growths = [0.070, 0.065, 0.060, 0.050, 0.045]
        margins = [0.320, 0.325, 0.330, 0.330, 0.330]
        tax_rate = 0.1561
        capex_pct = 0.0306

        result = calculate_forecast_schedule(
            base_revenue=base_rev,
            base_year=base_year,
            revenue_growth_rates=growths,
            operating_margins=margins,
            tax_rate=tax_rate,
            capex_pct_of_revenue=capex_pct,
        )

        # Schema & Provenance checks
        self.assertIsInstance(result, dict)
        self.assertEqual(result["provenance_mode"], "simplified_nopat_less_capex")
        self.assertEqual(result["base_revenue"], base_rev)
        self.assertEqual(result["base_year"], 2025)
        self.assertEqual(result["forecast_horizon_years"], 5)
        self.assertEqual(len(result["projected_fcfs"]), 5)
        self.assertEqual(len(result["forecast_schedule"]), 5)

        # Year 1 (FY2026) exact float math check
        # Rev: 416,161 * 1.07 = 445,292.27
        # EBIT: 445,292.27 * 0.32 = 142,493.5264
        # NOPAT: 142,493.5264 * (1 - 0.1561) = 120,250.28
        # CapEx: 445,292.27 * 0.0306 = 13,625.94
        # UFCF: 120,250.28 - 13,625.94 = 106,624.34
        y1 = result["forecast_schedule"][0]
        self.assertEqual(y1["projected_year"], 2026)
        self.assertAlmostEqual(y1["projected_revenue"], 445292.27, places=1)
        self.assertAlmostEqual(y1["projected_ebit"], 142493.53, places=1)
        self.assertAlmostEqual(y1["projected_capex"], 13625.94, places=1)
        self.assertAlmostEqual(y1["projected_unlevered_fcf"], 106624.34, places=1)
        self.assertEqual(result["projected_fcfs"][0], y1["projected_unlevered_fcf"])

        # 5-Year CAGR check: (551,577.29 / 416,161) ** (1/5) - 1 ≈ 5.79%
        self.assertAlmostEqual(result["revenue_cagr_pct"], 5.79, places=1)
        self.assertGreater(result["cumulative_5yr_fcf"], 500000.0)

        # Markdown table verification
        table = result["forecast_table_markdown"]
        self.assertIn("| Metric ($ Millions) | Base (FY2025) | FY2026E | FY2027E | FY2028E | FY2029E | FY2030E |", table)
        self.assertIn("| Revenue |", table)
        self.assertIn("| Operating Margin |", table)
        self.assertIn("| **Unlevered Free Cash Flow (UFCF)** |", table)

    def test_02_apple_fy2025_comprehensive_forecast(self):
        """
        Verify Option C 'comprehensive_line_item' mode:
        Includes D&A and Working Capital changes:
        UFCF = NOPAT + D&A - CapEx - ΔNWC
        """
        base_rev = 416161.0
        base_year = 2025
        growths = [0.07, 0.065, 0.06, 0.05, 0.045]
        margins = [0.32, 0.325, 0.33, 0.33, 0.33]
        tax_rate = 0.1561
        capex_pct = 0.0306
        depr_pct = 0.0281  # Apple 10-K D&A ~$11.7B
        nwc_pct = 0.010    # 1.0% of incremental revenue

        result = calculate_forecast_schedule(
            base_revenue=base_rev,
            base_year=base_year,
            revenue_growth_rates=growths,
            operating_margins=margins,
            tax_rate=tax_rate,
            capex_pct_of_revenue=capex_pct,
            depreciation_pct_of_revenue=depr_pct,
            nwc_change_pct_of_revenue=nwc_pct,
        )

        self.assertEqual(result["provenance_mode"], "comprehensive_line_item")

        # Year 1 (FY2026) exact float math check
        # Rev: 445,292.27 | delta_rev: 29,131.27
        # NOPAT: 120,250.28
        # D&A: 445,292.27 * 0.0281 = 12,512.71
        # CapEx: 445,292.27 * 0.0306 = 13,625.94
        # ΔNWC: 29,131.27 * 0.01 = 291.31
        # UFCF: 120,250.28 + 12,512.71 - 13,625.94 - 291.31 = 118,845.74
        y1 = result["forecast_schedule"][0]
        self.assertAlmostEqual(y1["projected_depreciation"], 12512.71, places=1)
        self.assertAlmostEqual(y1["projected_nwc_change"], 291.31, places=1)
        self.assertAlmostEqual(y1["projected_unlevered_fcf"], 118845.74, places=1)

        # Check table has Depreciation and Working Capital rows
        table = result["forecast_table_markdown"]
        self.assertIn("| Depreciation & Amortization |", table)
        self.assertIn("| Change in Working Capital (ΔNWC) |", table)

    def test_03_tesla_fy2025_high_reinvestment_forecast(self):
        """
        Verify Tesla FY2025 forecast with high capital expenditure intensity (~9% of revenue):
        Base Revenue: $94,827.0M
        CapEx: 9.0%
        Tax: 14.20%
        """
        result = calculate_forecast_schedule(
            base_revenue=94827.0,
            base_year=2025,
            revenue_growth_rates=[0.05, 0.10, 0.12, 0.15, 0.12],
            operating_margins=[0.06, 0.08, 0.10, 0.11, 0.12],
            tax_rate=0.1420,
            capex_pct_of_revenue=0.0899,
        )

        self.assertEqual(result["provenance_mode"], "simplified_nopat_less_capex")
        self.assertEqual(len(result["projected_fcfs"]), 5)

        # Verify increasing cash flows as operating margin expands from 6% to 12%
        fcfs = result["projected_fcfs"]
        self.assertLess(fcfs[0], fcfs[-1])
        self.assertGreater(result["revenue_cagr_pct"], 10.0)

    def test_04_nvidia_fy2026_fading_hypergrowth_forecast(self):
        """
        Verify NVIDIA FY2026 forecast with fading hyper-growth:
        Base Revenue: $215,938.0M
        Growth: 40% -> 25% -> 18% -> 14% -> 10%
        Margins: 60% -> 55%
        CapEx: 2.05%
        """
        result = calculate_forecast_schedule(
            base_revenue=215938.0,
            base_year=2026,
            revenue_growth_rates=[0.40, 0.25, 0.18, 0.14, 0.10],
            operating_margins=[0.60, 0.58, 0.56, 0.55, 0.55],
            tax_rate=0.1511,
            capex_pct_of_revenue=0.0205,
        )

        self.assertEqual(result["base_year"], 2026)
        self.assertEqual(result["forecast_schedule"][0]["projected_year"], 2027)
        self.assertEqual(result["forecast_schedule"][-1]["projected_year"], 2031)

        # Revenue compounds from $215.9B to >$500B
        final_rev = result["forecast_schedule"][-1]["projected_revenue"]
        self.assertGreater(final_rev, 500000.0)
        self.assertGreater(result["cumulative_5yr_fcf"], 1000000.0)  # >$1 Trillion in 5-year FCF

    def test_05_direct_handoff_into_dcf_valuation_engine(self):
        """
        End-to-End Pipeline Integration:
        calculate_forecast_schedule -> projected_fcfs -> calculate_dcf_with_sensitivity
        Validates seamless mathematical handoff without data reformatting.
        """
        # Step 1: Generate forward cash flows
        forecast = calculate_forecast_schedule(
            base_revenue=416161.0,
            base_year=2025,
            revenue_growth_rates=[0.07, 0.065, 0.06, 0.05, 0.045],
            operating_margins=[0.32, 0.325, 0.33, 0.33, 0.33],
            tax_rate=0.1561,
            capex_pct_of_revenue=0.0306,
        )
        projected_fcfs = forecast["projected_fcfs"]

        # Step 2: Pass directly into DCF valuation engine with Apple FY25 balance sheet
        wacc = 0.0953
        terminal_growth = 0.025
        net_debt = -33763.0  # Apple Net Cash Surplus
        diluted_shares = 15004.7

        dcf_result = calculate_dcf_with_sensitivity(
            projected_fcfs=projected_fcfs,
            wacc=wacc,
            terminal_growth_rate=terminal_growth,
            net_debt=net_debt,
            diluted_shares=diluted_shares,
            mid_year_convention=True,
        )

        self.assertGreater(dcf_result["enterprise_value"], 1500000.0)
        self.assertGreater(dcf_result["equity_value"], dcf_result["enterprise_value"])  # Cash surplus
        self.assertGreater(dcf_result["fair_value_per_share"], 100.0)
        self.assertIn("| WACC \\ Growth |", dcf_result["sensitivity_matrix_markdown"])

    def test_06_defensive_guards_and_auto_normalization(self):
        """Verify handling of percentage auto-normalization, invalid values, and bounds."""
        # 1. Whole percentages auto-normalization (e.g. 7.0% -> 0.07)
        res = calculate_forecast_schedule(
            base_revenue=1000.0,
            base_year=2025,
            revenue_growth_rates=[7.0, 6.0, 5.0, 4.0, 3.0],  # Passed as whole percentages
            operating_margins=[30.0, 30.0, 30.0, 30.0, 30.0],
            tax_rate=21.0,                                    # Passed as 21.0
            capex_pct_of_revenue=3.0,
        )
        self.assertAlmostEqual(res["forecast_schedule"][0]["projected_revenue"], 1070.0, places=1)
        self.assertAlmostEqual(res["tax_rate_pct"], 21.0, places=1)

        # 2. NaN / Inf rejection
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(
                base_revenue=float("nan"),
                base_year=2025,
                revenue_growth_rates=[0.05] * 5,
                operating_margins=[0.20] * 5,
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )

        with self.assertRaises(ValueError):
            calculate_forecast_schedule(
                base_revenue=1000.0,
                base_year=2025,
                revenue_growth_rates=[0.05, float("inf"), 0.05, 0.05, 0.05],
                operating_margins=[0.20] * 5,
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )

        # 3. Horizon length mismatch
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(
                base_revenue=1000.0,
                base_year=2025,
                revenue_growth_rates=[0.05, 0.05],       # 2 years
                operating_margins=[0.20, 0.20, 0.20],    # 3 years -> mismatch
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )

        # 4. Strictly positive base revenue
        with self.assertRaises(ValueError):
            calculate_forecast_schedule(
                base_revenue=-500.0,
                base_year=2025,
                revenue_growth_rates=[0.05] * 5,
                operating_margins=[0.20] * 5,
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )

    def test_07_langchain_tool_decorator(self):
        """Verify calculate_forecast_schedule_tool invocation via LangChain tool interface."""
        payload = {
            "base_revenue": 10000.0,
            "base_year": 2025,
            "revenue_growth_rates": [0.10, 0.08, 0.06, 0.05, 0.04],
            "operating_margins": [0.25, 0.25, 0.25, 0.25, 0.25],
            "tax_rate": 0.21,
            "capex_pct_of_revenue": 0.04,
        }

        tool_res = calculate_forecast_schedule_tool.invoke(payload)
        self.assertIsInstance(tool_res, dict)
        self.assertNotIn("error", tool_res)
        self.assertEqual(len(tool_res["projected_fcfs"]), 5)
        self.assertEqual(tool_res["provenance_mode"], "simplified_nopat_less_capex")

    def test_08_cagr_extreme_negative_growth_guard(self):
        """
        Verify Bug Fix 1:
        1. Growth rates <= -95% are rejected immediately to protect going-concern logic.
        2. Non-positive revenue prevents complex numbers in fractional power math (math.pow).
        """
        # Growth <= -0.95 raises ValueError
        with self.assertRaises(ValueError) as ctx:
            calculate_forecast_schedule(
                base_revenue=1000.0,
                base_year=2025,
                revenue_growth_rates=[-1.0, 0.05, 0.05, 0.05, 0.05],  # -100% collapse
                operating_margins=[0.20] * 5,
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )
        self.assertIn("cannot be <= -95.0%", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            calculate_forecast_schedule(
                base_revenue=1000.0,
                base_year=2025,
                revenue_growth_rates=[-0.98, 0.05, 0.05, 0.05, 0.05],
                operating_margins=[0.20] * 5,
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )
        self.assertIn("cannot be <= -95.0%", str(ctx.exception))

    def test_09_hypergrowth_and_distressed_margins_preserved(self):
        """
        Verify Bug Fix 2:
        1. Legitimate hyper-growth (e.g. 1.40 for 140%) is PRESERVED, not corrupted to 1.4%.
        2. Legitimate distressed negative margin (e.g. -1.50 for -150%) is PRESERVED, not corrupted to -1.5%.
        3. Post-normalization bounds reject absurd numbers (> 500% growth or < -200% margin).
        """
        # 140% hyper-growth (decimal 1.40) and -150% margin (decimal -1.50)
        res = calculate_forecast_schedule(
            base_revenue=1000.0,
            base_year=2025,
            revenue_growth_rates=[1.40, 0.80, 0.50, 0.30, 0.20],  # Decimals > 1.0!
            operating_margins=[-1.50, -0.80, -0.20, 0.10, 0.20], # Decimal margin < -1.0!
            tax_rate=0.21,
            capex_pct_of_revenue=0.05,
        )

        y1 = res["forecast_schedule"][0]
        # Revenue must be 1000 * 2.40 = 2400.0 (NOT 1014.0 from corrupt 1.4%!)
        self.assertAlmostEqual(y1["projected_revenue"], 2400.0, places=1)
        self.assertAlmostEqual(y1["projected_revenue_growth_pct"], 140.0, places=1)
        # EBIT must be 2400 * -1.50 = -3600.0 (NOT -36.0 from corrupt -1.5%!)
        self.assertAlmostEqual(y1["projected_ebit"], -3600.0, places=1)
        self.assertAlmostEqual(y1["projected_ebit_margin_pct"], -150.0, places=1)

        # Bounds: growth > 5.0 (500%) raises ValueError (e.g. 600.0% normalized to 6.0 > 5.0)
        with self.assertRaises(ValueError) as ctx:
            calculate_forecast_schedule(
                base_revenue=1000.0,
                base_year=2025,
                revenue_growth_rates=[600.0, 20.0, 20.0, 20.0, 20.0],  # 600% growth -> 6.0 > 5.0
                operating_margins=[0.20] * 5,
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )
        self.assertIn("exceeds maximum allowable threshold", str(ctx.exception))

        # Bounds: margin < -2.0 (-200%) raises ValueError (e.g. -250.0% normalized to -2.5 < -2.0)
        with self.assertRaises(ValueError) as ctx:
            calculate_forecast_schedule(
                base_revenue=1000.0,
                base_year=2025,
                revenue_growth_rates=[0.20] * 5,
                operating_margins=[-250.0, 10.0, 10.0, 10.0, 10.0],  # -250% margin -> -2.50 < -2.0
                tax_rate=0.21,
                capex_pct_of_revenue=0.03,
            )
        self.assertIn("below minimum allowable margin", str(ctx.exception))

    def test_10_required_params_reject_none(self):
        """
        Verify Bug Fix 3:
        Required parameters strictly reject None immediately rather than failing downstream.
        """
        valid_kwargs = {
            "base_revenue": 1000.0,
            "base_year": 2025,
            "revenue_growth_rates": [0.05] * 5,
            "operating_margins": [0.20] * 5,
            "tax_rate": 0.21,
            "capex_pct_of_revenue": 0.03,
        }

        # None for capex_pct_of_revenue
        bad_capex = dict(valid_kwargs, capex_pct_of_revenue=None)
        with self.assertRaises(ValueError) as ctx:
            calculate_forecast_schedule(**bad_capex)
        self.assertIn("capex_pct_of_revenue", str(ctx.exception))

        # None for tax_rate
        bad_tax = dict(valid_kwargs, tax_rate=None)
        with self.assertRaises(ValueError) as ctx:
            calculate_forecast_schedule(**bad_tax)
        self.assertIn("tax_rate", str(ctx.exception))

        # None in growth list entry
        bad_growth = dict(valid_kwargs, revenue_growth_rates=[0.05, None, 0.05, 0.05, 0.05])
        with self.assertRaises(ValueError) as ctx:
            calculate_forecast_schedule(**bad_growth)
        self.assertIn("revenue_growth_rates[1]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

