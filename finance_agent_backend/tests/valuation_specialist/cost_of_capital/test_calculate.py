"""cost_of_capital calculate: CAPM WACC, cost of debt and capital-structure weights.

Validates:
1. Deterministic calculation of Cost of Equity (Ke) via CAPM.
2. Dual market cap input resolution (direct market_cap vs share_price * diluted_shares).
3. Pre-tax and after-tax Cost of Debt (Kd) with strict provenance tagging:
   - Explicitly provided
   - Derived from 10-K interest expense
   - Institutional credit spread fallback
   - Zero-debt exemption
4. Strict IEEE 754 NaN and Inf rejection across all inputs.
5. Auto-normalization of rate inputs between percentage and decimal notation.
6. Real-world 10-K benchmarks across Tesla (TSLA FY25), Nvidia (NVDA FY26), and Apple (AAPL FY25).
7. Downstream pipeline handoff into calculate_dcf_with_sensitivity.
"""

import math
import unittest
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.calculate import (
    WACCInput,
    calculate_wacc,
)
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import WACCCalculationResult
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.calculate import calculate_dcf_with_sensitivity


class TestWACCTools(unittest.TestCase):
    """Comprehensive test suite for WACC calculation engine."""

    def test_01_market_cap_resolution_equivalence(self):
        """Verify passing market_cap directly vs (share_price * diluted_shares) gives identical WACC."""
        res_direct = calculate_wacc(
            beta=1.10,
            total_debt=98657.0,
            market_cap=3526104.5,
            risk_free_rate=0.042,
            equity_risk_premium=0.050,
            tax_rate=0.1561,
            cost_of_debt=0.042,
        )

        res_derived = calculate_wacc(
            beta=1.10,
            total_debt=98657.0,
            share_price=235.0,
            diluted_shares=15004.7,  # 235 * 15004.7 = 3,526,104.5
            risk_free_rate=0.042,
            equity_risk_premium=0.050,
            tax_rate=0.1561,
            cost_of_debt=0.042,
        )

        self.assertEqual(res_direct.wacc, res_derived.wacc)
        self.assertEqual(res_direct.cost_of_equity, res_derived.cost_of_equity)
        self.assertEqual(res_direct.market_cap, res_derived.market_cap)
        self.assertEqual(res_direct.total_capital, res_derived.total_capital)
        self.assertEqual(res_direct.equity_weight_pct, res_derived.equity_weight_pct)

    def test_02_cost_of_debt_provenance_tagging(self):
        """Verify transparent tagging for all 4 cost of debt paths."""
        # 1. Explicitly provided
        r1 = calculate_wacc(beta=1.0, total_debt=10000.0, market_cap=90000.0, cost_of_debt=0.05)
        self.assertEqual(r1.cost_of_debt_source, "explicit_provided")
        self.assertAlmostEqual(r1.cost_of_debt_pre_tax, 0.05, places=4)

        # 2. Derived from 10-K interest expense
        r2 = calculate_wacc(beta=1.0, total_debt=10000.0, market_cap=90000.0, interest_expense=450.0)
        self.assertEqual(r2.cost_of_debt_source, "derived_from_10k_interest_expense")
        self.assertAlmostEqual(r2.cost_of_debt_pre_tax, 0.045, places=4)

        # 3. Institutional credit spread fallback (Rf 4.2% + spread 1.25% = 5.45%)
        r3 = calculate_wacc(beta=1.0, total_debt=10000.0, market_cap=90000.0, risk_free_rate=0.042)
        self.assertEqual(r3.cost_of_debt_source, "institutional_credit_spread_fallback")
        self.assertAlmostEqual(r3.cost_of_debt_pre_tax, 0.0545, places=4)

        # 4. Zero debt exemption
        r4 = calculate_wacc(beta=1.0, total_debt=0.0, market_cap=90000.0)
        self.assertEqual(r4.cost_of_debt_source, "zero_debt_exemption")
        self.assertEqual(r4.cost_of_debt_pre_tax, 0.0)
        self.assertEqual(r4.debt_weight_pct, 0.0)
        self.assertEqual(r4.equity_weight_pct, 100.0)

    def test_03_zero_debt_pure_equity_hurdle(self):
        """Verify WACC collapses exactly to Cost of Equity when gross debt is zero."""
        beta = 1.30
        rf = 0.040
        erp = 0.055
        expected_ke = rf + (beta * erp)  # 0.04 + 0.0715 = 0.1115 (11.15%)

        result = calculate_wacc(
            beta=beta,
            total_debt=0.0,
            market_cap=50000.0,
            risk_free_rate=rf,
            equity_risk_premium=erp,
        )

        self.assertAlmostEqual(result.wacc, expected_ke, places=5)
        self.assertEqual(result.wacc_pct, 11.15)
        self.assertEqual(result.equity_weight_pct, 100.0)
        self.assertEqual(result.debt_weight_pct, 0.0)

    def test_04_strict_nan_inf_rejection(self):
        """Verify strict IEEE 754 NaN and Inf rejection."""
        invalid_inputs = [float("nan"), float("inf"), float("-inf")]

        for bad_val in invalid_inputs:
            with self.assertRaises(ValueError):
                calculate_wacc(beta=bad_val, total_debt=1000.0, market_cap=5000.0)

            with self.assertRaises(ValueError):
                calculate_wacc(beta=1.0, total_debt=bad_val, market_cap=5000.0)

            with self.assertRaises(ValueError):
                calculate_wacc(beta=1.0, total_debt=1000.0, market_cap=bad_val)

            with self.assertRaises(ValueError):
                calculate_wacc(beta=1.0, total_debt=1000.0, market_cap=5000.0, risk_free_rate=bad_val)

    def test_05_rate_auto_normalization(self):
        """Verify user passing percentage numbers like 4.2 or 21.0 gets normalized to decimals."""
        res_pct = calculate_wacc(
            beta=1.0,
            total_debt=10000.0,
            market_cap=90000.0,
            risk_free_rate=4.2,         # Passed 4.2 instead of 0.042
            equity_risk_premium=5.0,    # Passed 5.0 instead of 0.050
            tax_rate=21.0,              # Passed 21.0 instead of 0.21
            cost_of_debt=4.5,           # Passed 4.5 instead of 0.045
        )

        res_dec = calculate_wacc(
            beta=1.0,
            total_debt=10000.0,
            market_cap=90000.0,
            risk_free_rate=0.042,
            equity_risk_premium=0.050,
            tax_rate=0.21,
            cost_of_debt=0.045,
        )

        self.assertEqual(res_pct.wacc, res_dec.wacc)
        self.assertEqual(res_pct.cost_of_equity, res_dec.cost_of_equity)
        self.assertEqual(res_pct.cost_of_debt_pre_tax, res_dec.cost_of_debt_pre_tax)
        self.assertEqual(res_pct.effective_tax_rate, res_dec.effective_tax_rate)

    def test_06_real_world_tesla_fy2025_benchmark(self):
        """
        Verify real-world Tesla FY2025 benchmark:
        - Beta: 2.00 (high volatility)
        - Total Debt: $8,376M
        - Interest Expense: $338M (IS line item)
        - Tax Provision: $1,423M / Pretax $5,278M = 26.96%
        - Market Cap: ~$937,750M (3,751M shares @ $250)
        """
        res = calculate_wacc(
            beta=2.00,
            total_debt=8376.0,
            market_cap=937750.0,
            interest_expense=338.0,
            tax_rate=0.2696,
            risk_free_rate=0.042,
            equity_risk_premium=0.050,
        )

        # Ke = 4.2% + 2.0 * 5.0% = 14.20%
        self.assertAlmostEqual(res.cost_of_equity_pct, 14.20, places=2)
        # Kd = 338 / 8376 = 4.035%
        self.assertAlmostEqual(res.cost_of_debt_pre_tax_pct, 4.04, places=2)
        # After-tax Kd = 4.035% * (1 - 0.2696) = 2.947%
        self.assertAlmostEqual(res.cost_of_debt_after_tax_pct, 2.95, places=2)
        # Equity Weight ~ 99.11%
        self.assertGreater(res.equity_weight_pct, 99.0)
        # Final WACC ~ 14.10%
        self.assertAlmostEqual(res.wacc_pct, 14.10, places=2)
        self.assertEqual(res.cost_of_debt_source, "derived_from_10k_interest_expense")

    def test_07_real_world_nvidia_fy2026_benchmark(self):
        """
        Verify real-world Nvidia FY2026 benchmark:
        - Beta: 1.65
        - Total Debt: $8,468M ($999M short-term + $7,469M long-term)
        - Interest Expense: $259M (IS line item)
        - Tax Provision: $21,383M / Pretax $141,450M = 15.12%
        - Shares: 24,514M @ $130 = $3,186,820M
        """
        res = calculate_wacc(
            beta=1.65,
            total_debt=8468.0,
            share_price=130.0,
            diluted_shares=24514.0,
            interest_expense=259.0,
            tax_rate=0.1512,
            risk_free_rate=0.042,
            equity_risk_premium=0.050,
        )

        # Ke = 4.2% + 1.65 * 5.0% = 12.45%
        self.assertAlmostEqual(res.cost_of_equity_pct, 12.45, places=2)
        # Kd = 259 / 8468 = 3.06% is a stale coupon below Rf - 1pp (3.20%): marginal Kd = Rf + 1.25% = 5.45%
        self.assertEqual(res.cost_of_debt_source, "interest_expense_out_of_band_fallback")
        self.assertAlmostEqual(res.cost_of_debt_pre_tax_pct, 5.45, places=2)
        self.assertIn("3.06%", res.cost_of_debt_note)
        # After-tax Kd = 5.45% * (1 - 0.1512) = 4.63%; E = 3,186,820, D = 8,468 => WACC 12.43%
        self.assertAlmostEqual(res.cost_of_debt_after_tax_pct, 4.63, places=2)
        self.assertAlmostEqual(res.wacc_pct, 12.43, places=2)

    def test_08_real_world_apple_fy2025_benchmark(self):
        """
        Verify real-world Apple FY2025 benchmark:
        - Beta: 1.10
        - Total Debt: $98,657M
        - Effective Tax Rate: 15.61%
        - Shares: 15,004.7M @ $235 = $3,526,104.5M
        - Cost of Debt: 4.20% (from Note 8 Debt table)
        """
        res = calculate_wacc(
            beta=1.10,
            total_debt=98657.0,
            share_price=235.0,
            diluted_shares=15004.7,
            cost_of_debt=0.042,
            tax_rate=0.1561,
            risk_free_rate=0.042,
            equity_risk_premium=0.050,
        )

        # Ke = 4.2% + 1.10 * 5.0% = 9.70%
        self.assertAlmostEqual(res.cost_of_equity_pct, 9.70, places=2)
        # After-tax Kd = 4.2% * (1 - 0.1561) = 3.544%
        self.assertAlmostEqual(res.cost_of_debt_after_tax_pct, 3.54, places=2)
        # Equity Weight = 3,526,104.5 / (3,526,104.5 + 98,657) = 97.28%
        self.assertAlmostEqual(res.equity_weight_pct, 97.28, places=2)
        # Final WACC = (0.9728 * 9.70%) + (0.0272 * 3.544%) = 9.53%
        self.assertAlmostEqual(res.wacc_pct, 9.53, places=2)
        self.assertEqual(res.cost_of_debt_source, "explicit_provided")
        self.assertIn("CAPM Blended WACC", res.formula_breakdown_markdown)

    def test_09_dcf_pipeline_handoff_integration(self):
        """Verify seamless pipeline handoff: wacc_calculator output feeds directly into calculate_dcf."""
        # 1. Calculate WACC for Apple
        wacc_out = calculate_wacc(
            beta=1.10,
            total_debt=98657.0,
            share_price=235.0,
            diluted_shares=15004.7,
            cost_of_debt=0.042,
            tax_rate=0.1561,
        )

        # 2. Pass resulting wacc directly into DCF valuation engine
        dcf_out = calculate_dcf_with_sensitivity(
            projected_fcfs=[110000.0, 118000.0, 126000.0, 134000.0, 142000.0],
            wacc=wacc_out.wacc,  # Direct handoff!
            terminal_growth_rate=0.025,
            net_debt=-33763.0,      # Cash surplus for Apple
            diluted_shares=15004.7,
            mid_year_convention=True,
        )

        self.assertGreater(dcf_out.enterprise_value, 0)
        self.assertGreater(dcf_out.equity_value, dcf_out.enterprise_value)  # Cash cushion
        self.assertGreater(dcf_out.fair_value_per_share, 100.0)
        self.assertIn("| WACC \\ Growth |", dcf_out.sensitivity_matrix_markdown)

    def test_11_beta_boundary_enforcement(self):
        """Verify strict economic boundary on beta (0.0 < beta <= 5.0)."""
        # Beta <= 0
        with self.assertRaises(ValueError) as ctx_zero:
            calculate_wacc(beta=0.0, total_debt=1000.0, market_cap=10000.0)
        self.assertIn("realistic equity market bounds", str(ctx_zero.exception).lower())

        with self.assertRaises(ValueError) as ctx_neg:
            calculate_wacc(beta=-0.5, total_debt=1000.0, market_cap=10000.0)
        self.assertIn("realistic equity market bounds", str(ctx_neg.exception).lower())

        # Beta > 5.0 (e.g. extracted P/E multiple of 45 or 50)
        with self.assertRaises(ValueError) as ctx_high:
            calculate_wacc(beta=50.0, total_debt=1000.0, market_cap=10000.0)
        self.assertIn("realistic equity market bounds", str(ctx_high.exception).lower())

    def test_12_cost_of_debt_boundary_enforcement(self):
        """Out-of-band derived Kd falls back to Rf + spread (no raise); explicit Kd > 30% still raises."""
        res = calculate_wacc(beta=1.1, total_debt=500.0, market_cap=50000.0, interest_expense=300.0)  # 60%
        self.assertEqual(res.cost_of_debt_source, "interest_expense_out_of_band_fallback")
        self.assertAlmostEqual(res.cost_of_debt_pre_tax, 0.042 + 0.0125, places=6)
        self.assertIn("60.00%", res.cost_of_debt_note)

        with self.assertRaises(ValueError) as ctx_explicit:
            calculate_wacc(beta=1.1, total_debt=5000.0, market_cap=50000.0, cost_of_debt=0.35)
        self.assertIn("exceeds realistic borrowing limits", str(ctx_explicit.exception))

    def test_13_kd_uses_average_debt(self):
        res = calculate_wacc(
            beta=1.0, total_debt=12000.0, prior_total_debt=8000.0, market_cap=90000.0, interest_expense=500.0,
        )
        self.assertEqual(res.cost_of_debt_source, "derived_from_10k_interest_expense")
        self.assertAlmostEqual(res.cost_of_debt_pre_tax, 500.0 / 10000.0, places=6)

    def test_14_blume_adjustment_only_when_requested(self):
        raw = calculate_wacc(beta=1.6, total_debt=0.0, market_cap=1000.0, risk_free_rate=0.04, equity_risk_premium=0.05)
        adj = calculate_wacc(
            beta=1.6, total_debt=0.0, market_cap=1000.0, risk_free_rate=0.04, equity_risk_premium=0.05,
            apply_blume_adjustment=True, beta_source="yfinance_raw",
        )
        self.assertEqual(raw.beta, 1.6)
        self.assertAlmostEqual(adj.beta, 0.67 * 1.6 + 0.33, places=4)
        self.assertEqual(adj.beta_raw, 1.6)
        self.assertAlmostEqual(adj.cost_of_equity, 0.04 + (0.67 * 1.6 + 0.33) * 0.05, places=6)
        self.assertIn("Blume-adjusted from raw 1.60", adj.formula_breakdown_markdown)

    def test_15_override_keeps_capm_components(self):
        res = calculate_wacc(
            beta=1.2, total_debt=10000.0, market_cap=90000.0, cost_of_debt=0.05, wacc_override=0.11,
        )
        base = calculate_wacc(beta=1.2, total_debt=10000.0, market_cap=90000.0, cost_of_debt=0.05)
        self.assertEqual(res.wacc, 0.11)
        self.assertEqual(res.wacc_source, "user_override")
        self.assertEqual(res.capm_wacc_pct, base.wacc_pct)
        self.assertEqual(res.debt_weight_pct, 10.0)
        self.assertEqual(res.cost_of_equity, base.cost_of_equity)

    def test_16_missing_equity_value_weights_all_equity(self):
        res = calculate_wacc(beta=1.0, total_debt=5000.0, risk_free_rate=0.04, equity_risk_premium=0.05)
        self.assertEqual(res.capital_structure_source, "all_equity_no_equity_value")
        self.assertEqual(res.equity_weight_pct, 100.0)
        self.assertAlmostEqual(res.wacc, 0.09, places=6)
        self.assertIsNone(res.market_cap)

    def test_17_sources_are_reported(self):
        res = calculate_wacc(
            beta=1.0, total_debt=0.0, market_cap=1000.0, risk_free_rate=0.043,
            risk_free_rate_source="us_10y_treasury_live", equity_value_source="book_equity_fallback",
        )
        self.assertEqual(res.risk_free_rate_source, "us_10y_treasury_live")
        self.assertEqual(res.equity_risk_premium_source, "house_assumption_config")
        self.assertEqual(res.capital_structure_source, "book_equity_fallback")
        self.assertNotIn("Damodaran", res.formula_breakdown_markdown)

if __name__ == "__main__":
    unittest.main()
