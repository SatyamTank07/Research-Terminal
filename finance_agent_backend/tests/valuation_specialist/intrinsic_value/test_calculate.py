"""intrinsic_value calculate: the DCF engine and its equity bridge.

Validates the deterministic DCF calculation engine, Gordon Growth discounting,
institutional Mid-Year Discounting vs. Year-End conventions, negative net debt (cash cushion),
economic reasonableness boundary validations (e.g. g <= 0.05), NaN/Inf hardening across all inputs,
whole-percentage WACC detection, and the EV-to-equity bridge.
"""

import unittest
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.calculate import (
    calculate_dcf,
    dcf_fair_value_per_share,
)
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.thresholds import (
    TERMINAL_EV_EBITDA_BAND,
    TERMINAL_EV_NOPAT_BAND,
)


class TestDCFTools(unittest.TestCase):
    """Test suite for deterministic DCF valuation and sensitivity engine."""

    def test_01_standard_5yr_dcf_exact_math_year_end(self):
        """Verify mathematical precision against analytical benchmark under Year-End convention."""
        fcfs = [100.0, 110.0, 120.0, 130.0, 140.0]
        wacc = 0.10
        g = 0.02
        net_debt = 50.0
        shares = 10.0

        result = calculate_dcf(
            projected_fcfs=fcfs,
            wacc=wacc,
            terminal_growth_rate=g,
            net_debt=net_debt,
            diluted_shares=shares,
            mid_year_convention=False,
        )

        # Mathematical verification under year-end convention:
        # PV FCFs = 100/1.1 + 110/1.21 + 120/1.331 + 130/1.4641 + 140/1.61051 = 447.70
        self.assertAlmostEqual(result.pv_explicit_fcfs, 447.70, places=1)

        # TV = 140 * 1.02 / 0.08 = 1785.0; PV(TV) = 1785 / 1.61051 = 1108.35
        self.assertAlmostEqual(result.pv_terminal_value, 1108.35, places=1)

        # EV = 447.70 + 1108.35 = 1556.05
        self.assertAlmostEqual(result.enterprise_value, 1556.05, places=1)

        # Equity Value = 1556.05 - 50.0 = 1506.05
        self.assertAlmostEqual(result.equity_value, 1506.05, places=1)

        # Fair Value Per Share = 1506.05 / 10 = 150.60
        self.assertAlmostEqual(result.fair_value_per_share, 150.60, places=1)
        self.assertEqual(result.discounting_convention, "year_end")

    def test_01b_mid_year_convention_uplift(self):
        """Verify institutional Mid-Year Discounting provides expected half-year cash flow uplift."""
        fcfs = [100.0, 110.0, 120.0, 130.0, 140.0]
        wacc = 0.10
        g = 0.02
        net_debt = 50.0
        shares = 10.0

        res_year_end = calculate_dcf(fcfs, wacc, g, net_debt, shares, mid_year_convention=False)
        res_mid_year = calculate_dcf(fcfs, wacc, g, net_debt, shares, mid_year_convention=True)

        # Explicit cash flows under mid-year convention are discounted at (t - 0.5), so PV must be higher
        self.assertGreater(res_mid_year.pv_explicit_fcfs, res_year_end.pv_explicit_fcfs)

        # Ratio of mid-year to year-end PV of explicit cash flows should equal sqrt(1 + wacc) = sqrt(1.10) ~ 1.0488
        uplift_ratio = res_mid_year.pv_explicit_fcfs / res_year_end.pv_explicit_fcfs
        self.assertAlmostEqual(uplift_ratio, (1.0 + wacc) ** 0.5, places=2)

        # Resulting fair value per share is higher under mid-year convention
        self.assertGreater(res_mid_year.fair_value_per_share, res_year_end.fair_value_per_share)
        self.assertIn("mid_year", res_mid_year.discounting_convention)

    def test_02_negative_net_debt_cash_cushion(self):
        """Verify negative net debt (cash surplus) correctly adds to equity value."""
        fcfs = [100.0, 100.0, 100.0, 100.0, 100.0]
        wacc = 0.08
        g = 0.02
        positive_debt = 200.0
        negative_debt = -200.0  # $200M net cash surplus
        shares = 10.0

        res_pos = calculate_dcf(fcfs, wacc, g, positive_debt, shares)
        res_neg = calculate_dcf(fcfs, wacc, g, negative_debt, shares)

        # Enterprise values must be identical (operational cash flows unchanged)
        self.assertEqual(res_pos.enterprise_value, res_neg.enterprise_value)

        # Equity value with net cash should be exactly $400M higher ($200M debt vs -$200M cash)
        diff = res_neg.equity_value - res_pos.equity_value
        self.assertAlmostEqual(diff, 400.0, places=1)

        # Fair value per share must be $40.0 higher ($400M / 10M shares)
        self.assertAlmostEqual(res_neg.fair_value_per_share - res_pos.fair_value_per_share, 40.0, places=1)

    def test_03_boundary_and_validation_errors(self):
        """Verify strict error raising for invalid economic parameters."""
        valid_fcfs = [100.0, 110.0, 120.0, 130.0, 140.0]

        # 1. Growth >= WACC (Gordon Growth divergence)
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.08, terminal_growth_rate=0.08, net_debt=0, diluted_shares=10)

        # 2. Economic bounds: g > 0.05 (unrealistically high perpetual GDP growth)
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.09, terminal_growth_rate=0.06, net_debt=0, diluted_shares=10)

        # 3. Economic bounds: g < -0.02 (unrealistically negative)
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.08, terminal_growth_rate=-0.05, net_debt=0, diluted_shares=10)

        # 4. Non-positive shares
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.08, terminal_growth_rate=0.02, net_debt=0, diluted_shares=0)

        # 5. Non-positive WACC
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.0, terminal_growth_rate=0.02, net_debt=0, diluted_shares=10)

        # 6. Whole-percentage WACC handoff error (e.g. 8.5 instead of 0.085)
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=8.5, terminal_growth_rate=0.02, net_debt=0, diluted_shares=10)

        # 7. Unrealistic WACC (> 50%)
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.55, terminal_growth_rate=0.02, net_debt=0, diluted_shares=10)

        # 8. Empty FCFs
        with self.assertRaises(ValueError):
            calculate_dcf([], wacc=0.08, terminal_growth_rate=0.02, net_debt=0, diluted_shares=10)

    def test_03b_nan_inf_validation_hardening(self):
        """Verify NaN/Inf across all inputs raises ValueError rather than returning corrupted floats."""
        valid_fcfs = [100.0, 110.0, 120.0, 130.0, 140.0]

        # NaN in WACC
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=float("nan"), terminal_growth_rate=0.02, net_debt=0, diluted_shares=10)

        # NaN in terminal_growth_rate
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.08, terminal_growth_rate=float("nan"), net_debt=0, diluted_shares=10)

        # NaN in net_debt
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.08, terminal_growth_rate=0.02, net_debt=float("nan"), diluted_shares=10)

        # NaN in diluted_shares
        with self.assertRaises(ValueError):
            calculate_dcf(valid_fcfs, wacc=0.08, terminal_growth_rate=0.02, net_debt=0, diluted_shares=float("nan"))

        # Inf in FCFs
        with self.assertRaises(ValueError):
            calculate_dcf([100.0, float("inf")], wacc=0.08, terminal_growth_rate=0.02, net_debt=0, diluted_shares=10)

    def test_07_mid_year_terminal_value_discounted_at_n_minus_half(self):
        """Perpetuity TV under mid-year convention is discounted at N - 0.5 (Rosenbaum & Pearl / McKinsey)."""
        fcfs = [100.0, 110.0, 120.0, 130.0, 140.0]
        mid = calculate_dcf(fcfs, 0.10, 0.02, 0.0, 10.0, mid_year_convention=True)
        end = calculate_dcf(fcfs, 0.10, 0.02, 0.0, 10.0, mid_year_convention=False)
        # TV = 140 * 1.02 / 0.08 = 1785; PV = 1785 / 1.1^4.5 = 1162.44
        self.assertAlmostEqual(mid.terminal_value, 1785.0, places=2)
        self.assertAlmostEqual(mid.pv_terminal_value, 1785.0 / 1.1 ** 4.5, places=1)
        self.assertAlmostEqual(mid.pv_terminal_value / end.pv_terminal_value, 1.1 ** 0.5, places=4)
        self.assertEqual(mid.terminal_method, "last_fcf_growth")

    def test_08_value_driver_terminal_cash_flow(self):
        fcfs = [100.0, 110.0, 120.0, 130.0, 140.0]
        res = calculate_dcf(fcfs, 0.10, 0.02, 0.0, 10.0, terminal_nopat=200.0, ronic=0.20)
        # NOPAT x (1+g) x (1 - g/RONIC) = 200 x 1.02 x 0.9 = 183.6; TV = 183.6 / 0.08 = 2295
        self.assertEqual(res.terminal_method, "value_driver")
        self.assertAlmostEqual(res.terminal_fcf, 183.6, places=2)
        self.assertAlmostEqual(res.terminal_value, 2295.0, places=1)
        self.assertEqual(res.ronic, 0.2)

        # RONIC = WACC makes growth value-neutral: TV = NOPAT_{N+1} / WACC
        neutral = calculate_dcf(fcfs, 0.10, 0.02, 0.0, 10.0, terminal_nopat=200.0, ronic=0.10)
        self.assertAlmostEqual(neutral.terminal_value, 200.0 * 1.02 / 0.10, places=1)

        # Non-positive NOPAT falls back to growing the final UFCF, with a warning
        loss = calculate_dcf(fcfs, 0.10, 0.02, 0.0, 10.0, terminal_nopat=-5.0, ronic=0.2)
        self.assertEqual(loss.terminal_method, "last_fcf_growth")
        self.assertTrue(any("NOPAT is not positive" in w for w in loss.warnings))

    def test_09_bridge_deducts_nci_and_rolls_forward(self):
        fcfs = [100.0, 110.0, 120.0, 130.0, 140.0]
        plain = calculate_dcf(fcfs, 0.10, 0.02, 50.0, 10.0)
        with_nci = calculate_dcf(fcfs, 0.10, 0.02, 50.0, 10.0, noncontrolling_interests=30.0)
        self.assertAlmostEqual(plain.equity_value - with_nci.equity_value, 30.0, places=1)
        self.assertIn("Less: noncontrolling interests | $30.0M", with_nci.equity_bridge_markdown)
        self.assertIn("Less: net debt | $50.0M", with_nci.equity_bridge_markdown)
        cash_rich = calculate_dcf(fcfs, 0.10, 0.02, -50.0, 10.0)
        self.assertIn("Plus: net cash | $50.0M", cash_rich.equity_bridge_markdown)

        # EV rolls forward at WACC; net debt and NCI at the risk-free rate (not the equity value at Ke).
        rolled = calculate_dcf(
            fcfs, 0.10, 0.02, 50.0, 10.0, roll_forward_years=0.5, risk_free_rate=0.04, noncontrolling_interests=30.0,
        )
        self.assertEqual(rolled.enterprise_value, plain.enterprise_value)
        expected = plain.enterprise_value * 1.10 ** 0.5 - (50.0 + 30.0) * 1.04 ** 0.5
        self.assertAlmostEqual(rolled.equity_value, expected, places=1)
        self.assertAlmostEqual(rolled.roll_forward_factor, 1.10 ** 0.5, places=5)
        self.assertAlmostEqual(rolled.claims_roll_forward_factor, 1.04 ** 0.5, places=5)
        self.assertIn("Enterprise value rolled forward 0.50 yrs @ WACC 10.00%", rolled.equity_bridge_markdown)
        self.assertIn("Net debt and NCI rolled forward @ Rf 4.00%", rolled.equity_bridge_markdown)

        # A net-cash company: the cash pile earns Rf, so it no longer compounds at the cost of equity.
        cash = calculate_dcf(fcfs, 0.10, 0.02, -500.0, 10.0, roll_forward_years=1.0, risk_free_rate=0.04)
        self.assertAlmostEqual(cash.equity_value, cash.enterprise_value * 1.10 + 500.0 * 1.04, places=1)

    def test_10_sanity_warnings_do_not_raise(self):
        concentrated = calculate_dcf([10.0, 10.0, 10.0], 0.04, 0.035, 0.0, 1.0, risk_free_rate=0.03)
        joined = " ".join(concentrated.warnings)
        self.assertIn("of enterprise value", joined)
        self.assertIn("exceeds the risk-free rate", joined)

        negative = calculate_dcf([50.0, 20.0, -40.0], 0.09, 0.02, 0.0, 10.0)
        self.assertLess(negative.terminal_value, 0)
        self.assertTrue(any("Terminal value is not positive" in w for w in negative.warnings))

        stale = calculate_dcf([100.0] * 5, 0.09, 0.02, 0.0, 10.0, roll_forward_years=2.0)
        self.assertTrue(any("stale" in w for w in stale.warnings))

    def test_11_implied_terminal_ev_ebitda(self):
        res = calculate_dcf(
            [100.0, 110.0, 120.0, 130.0, 140.0], 0.0937, 0.025, -20.0, 10.0,
            terminal_nopat=150.0, ronic=0.25, noncontrolling_interests=5.0, roll_forward_years=0.3,
            risk_free_rate=0.04, terminal_ebitda=180.0,
        )
        self.assertAlmostEqual(res.implied_terminal_ev_ebitda, round(res.terminal_value / 180.0, 2))

    def test_06_real_world_aapl_valuation(self):
        """Verify DCF calculation using real Apple FY2025 audited 10-K figures."""
        # Audited AAPL FY2025: Net cash surplus of ~$33.8B (-$33,763M), Shares ~15,005M
        # 5-Year projected UFCFs based on 4% moderate growth from FY2025 $98.8B FCF
        aapl_fcfs = [102717.0, 106826.0, 111099.0, 115543.0, 120165.0]
        wacc = 0.088  # 8.8% Wall Street consensus WACC for AAPL
        g = 0.025     # 2.5% long-term GDP growth
        net_debt = -33763.0  # Net Cash
        shares = 15004.7

        result = calculate_dcf(aapl_fcfs, wacc, g, net_debt, shares, mid_year_convention=True)

        # Implied fair value should be in sound institutional range ($100 - $260)
        self.assertGreater(result.fair_value_per_share, 100.0)
        self.assertLess(result.fair_value_per_share, 260.0)
        self.assertGreater(result.enterprise_value, 1_500_000.0)  # > $1.5 Trillion EV


class TestImpliedMultiples(unittest.TestCase):
    FCFS = [100.0, 110.0, 120.0, 130.0, 140.0]

    def test_low_implied_multiples_are_warned(self):
        # Heavy final-year reinvestment: UFCF far below NOPAT, last-FCF TV => ~2x NOPAT, ~1.5x EBITDA
        res = calculate_dcf(
            [10.0] * 5, 0.12, 0.025, 0.0, 10.0, terminal_nopat=60.0, terminal_ebitda=80.0,
        )
        self.assertEqual(res.terminal_method, "last_fcf_growth")
        self.assertLess(res.implied_terminal_ev_ebitda, TERMINAL_EV_EBITDA_BAND[0])
        self.assertLess(res.implied_terminal_ev_nopat, TERMINAL_EV_NOPAT_BAND[0])
        joined = " ".join(res.warnings)
        self.assertIn("EV/EBITDA", joined)
        self.assertIn("EV/NOPAT", joined)
        self.assertIn("below", joined)

    def test_high_implied_multiple_is_warned(self):
        res = calculate_dcf(
            [100.0] * 5, 0.06, 0.035, 0.0, 10.0, terminal_nopat=100.0, ronic=0.30, terminal_ebitda=60.0,
        )
        self.assertGreater(res.implied_terminal_ev_ebitda, TERMINAL_EV_EBITDA_BAND[1])
        self.assertIn("above", " ".join(res.warnings))

    def test_plausible_multiples_are_silent(self):
        res = calculate_dcf(
            self.FCFS, 0.09, 0.025, 0.0, 10.0, terminal_nopat=180.0, ronic=0.15, terminal_ebitda=260.0,
        )
        self.assertTrue(TERMINAL_EV_NOPAT_BAND[0] <= res.implied_terminal_ev_nopat <= TERMINAL_EV_NOPAT_BAND[1])
        self.assertFalse(any("plausibility band" in w for w in res.warnings))

    def test_multiples_none_when_terminal_value_not_positive(self):
        res = calculate_dcf([50.0, 20.0, -40.0], 0.09, 0.02, 0.0, 10.0,
                                             terminal_nopat=-5.0, ronic=0.2, terminal_ebitda=10.0)
        self.assertIsNone(res.implied_terminal_ev_ebitda)
        self.assertIsNone(res.implied_terminal_ev_nopat)

class TestDcfFairValuePerShare(unittest.TestCase):
    def test_matches_headline_engine(self):
        kwargs = dict(terminal_nopat=180.0, ronic=0.15, noncontrolling_interests=20.0,
                      roll_forward_years=0.75, risk_free_rate=0.04)
        full = calculate_dcf([100.0, 110.0, 120.0], 0.09, 0.025, 50.0, 10.0, **kwargs)
        quick = dcf_fair_value_per_share([100.0, 110.0, 120.0], 0.09, 0.025, 50.0, 10.0, **kwargs)
        self.assertAlmostEqual(quick, full.fair_value_per_share, places=2)
        last = dcf_fair_value_per_share([100.0, 110.0, 120.0], 0.09, 0.025, -50.0, 10.0)
        self.assertAlmostEqual(
            last, calculate_dcf([100.0, 110.0, 120.0], 0.09, 0.025, -50.0, 10.0).fair_value_per_share,
            places=2,
        )

    def test_never_raises(self):
        for args in (
            ([], 0.09, 0.02, 0.0, 10.0),
            ([100.0], 0.02, 0.03, 0.0, 10.0),
            ([100.0], 0.09, 0.02, 0.0, 0.0),
            ([float("nan")], 0.09, 0.02, 0.0, 10.0),
            ([100.0], 8.5, 0.02, 0.0, 10.0),
            (["x"], 0.09, 0.02, 0.0, 10.0),
        ):
            self.assertIsNone(dcf_fair_value_per_share(*args), args)


if __name__ == "__main__":
    unittest.main()
