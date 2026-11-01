"""Tests for build_valuation's reverse DCF, implied-multiple checks, RONIC resolution and company-profile edges."""

import unittest

from app.agents.shared.prompts import render_prompt
from app.agents.specialized.valuation_specialist import ValuationSpecialistAgent
from app.agents.tools.dcf_tools import dcf_fair_value_per_share
from app.agents.tools.forecast_model import project_with_overrides
from app.agents.specialized.valuation_specialist.context import MarketInputs
from app.agents.specialized.valuation_specialist.assemble import (
    REVERSE_DCF_MAX_ITERATIONS,
    REVERSE_DCF_SCAN_POINTS,
    build_valuation,
    solve_reverse_dcf,
)
from tests.forecast_fixtures import (
    make_aapl_audit,
    make_heavy_capex_audit,
    make_loss_audit,
    make_shrinking_audit,
    make_forecast_output,
)
from tests.valuation_fixtures import AAPL_BALANCE_SHEET_DATE, VALUATION_DATE, make_market, make_valuation_output

REVERSE_STATUSES = {
    "solved", "price_above_range", "price_below_range", "no_solution", "price_unavailable", "forecast_unavailable",
}


def _value(audit, forecast, market, ticker="TEST"):
    return build_valuation(
        ticker, 2025, audit, forecast, market, terminal_growth_rate=0.025,
        valuation_date=VALUATION_DATE, balance_sheet_date=AAPL_BALANCE_SHEET_DATE,
    )



def nopat_base(forecast, v):
    """Headline terminal NOPAT: final-year EBIT re-taxed at the marginal rate."""
    return round(forecast.forecast_schedule[-1].projected_ebit * (1.0 - v.terminal_tax_rate_pct / 100.0), 2)


class TestSolveReverseDcf(unittest.TestCase):
    def test_linear_root(self):
        root, status = solve_reverse_dcf(lambda g: 100.0 + 1000.0 * g, 150.0)
        self.assertEqual(status, "solved")
        self.assertAlmostEqual(root, 0.05, places=4)

    def test_picks_bracket_nearest_anchor(self):
        def value(g):  # two crossings of 0: at -0.1 and 0.3
            return (g + 0.1) * (g - 0.3)
        self.assertAlmostEqual(solve_reverse_dcf(value, 0.0, anchor=0.25)[0], 0.3, places=4)
        self.assertAlmostEqual(solve_reverse_dcf(value, 0.0, anchor=-0.05)[0], -0.1, places=4)

    def test_no_root_reports_side(self):
        self.assertEqual(solve_reverse_dcf(lambda g: 10.0, 50.0), (None, "price_above_range"))
        self.assertEqual(solve_reverse_dcf(lambda g: 100.0, 50.0), (None, "price_below_range"))

    def test_undefined_or_raising_value_fn(self):
        self.assertEqual(solve_reverse_dcf(lambda g: None, 50.0), (None, "no_solution"))

        def boom(g):
            raise ValueError("bad input")
        self.assertEqual(solve_reverse_dcf(boom, 50.0), (None, "no_solution"))
        self.assertEqual(solve_reverse_dcf(lambda g: float("nan"), 50.0), (None, "no_solution"))

    def test_partially_undefined_curve_still_solves(self):
        root, status = solve_reverse_dcf(lambda g: None if g < 0 else 100.0 + 1000.0 * g, 200.0)
        self.assertEqual(status, "solved")
        self.assertAlmostEqual(root, 0.10, places=4)

    def test_bounded_evaluations(self):
        calls = []

        def value(g):
            calls.append(g)
            return 100.0 + 1000.0 * g
        solve_reverse_dcf(value, 123.456)
        self.assertLessEqual(len(calls), REVERSE_DCF_SCAN_POINTS + REVERSE_DCF_MAX_ITERATIONS)


class TestReverseDcfInBuildValuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = make_aapl_audit()
        cls.forecast = make_forecast_output()

    def test_market_implied_growth_reprices_to_the_share_price(self):
        v = _value(self.audit, self.forecast, make_market(share_price=235.0), "AAPL")
        self.assertEqual(v.reverse_dcf_status, "solved")
        self.assertEqual(v.reverse_dcf_horizon_years, 5)
        self.assertEqual(v.modeled_revenue_cagr_pct, self.forecast.revenue_cagr_pct)
        self.assertGreater(v.market_implied_growth_pct, v.modeled_revenue_cagr_pct)  # price is above fair value
        g = v.market_implied_growth_pct / 100.0
        fcfs, nopat = project_with_overrides(
            self.forecast.assumptions, self.forecast.applied_drivers, 0.025, growth=[g] * 5,
            terminal_tax=v.terminal_tax_rate_pct / 100.0,
        )
        w = v.wacc_audit
        repriced = dcf_fair_value_per_share(
            fcfs, w.wacc, 0.025, v.net_debt, v.diluted_shares, terminal_nopat=nopat, ronic=v.ronic_used_pct / 100.0,
            roll_forward_years=v.roll_forward_years, risk_free_rate=w.risk_free_rate_pct / 100.0,
        )
        self.assertAlmostEqual(repriced, 235.0, delta=0.5)

        # The market-implied discount rate reprices the unchanged base case to the same price.
        self.assertIsNotNone(v.market_implied_wacc_pct)
        self.assertLess(v.market_implied_wacc_pct, w.wacc_pct)  # price above fair value -> lower required return
        at_implied = dcf_fair_value_per_share(
            list(v.projected_fcfs), v.market_implied_wacc_pct / 100.0, 0.025, v.net_debt, v.diluted_shares,
            terminal_nopat=nopat_base(self.forecast, v), ronic=v.ronic_used_pct / 100.0,
            roll_forward_years=v.roll_forward_years, risk_free_rate=w.risk_free_rate_pct / 100.0,
        )
        self.assertAlmostEqual(at_implied, 235.0, delta=0.5)
        self.assertIn("discount rate versus the", v.reverse_dcf_summary)
        self.assertIn("market implies", v.reverse_dcf_summary)

        # Companion solve: uniform margin shift at modeled growth reprices to the same price.
        self.assertGreater(v.market_implied_margin_shift_pp, 0)
        self.assertEqual(v.modeled_final_margin_pct, round(self.forecast.applied_drivers.operating_margin[-1] * 100, 2))
        shift = v.market_implied_margin_shift_pp / 100.0
        margins = [m + shift for m in self.forecast.applied_drivers.operating_margin]
        fcfs, nopat = project_with_overrides(self.forecast.assumptions, self.forecast.applied_drivers, 0.025,
                                             margins=margins, terminal_tax=v.terminal_tax_rate_pct / 100.0)
        repriced = dcf_fair_value_per_share(
            fcfs, w.wacc, 0.025, v.net_debt, v.diluted_shares, terminal_nopat=nopat, ronic=v.ronic_used_pct / 100.0,
            roll_forward_years=v.roll_forward_years, risk_free_rate=w.risk_free_rate_pct / 100.0,
        )
        self.assertAlmostEqual(repriced, 235.0, delta=0.5)
        self.assertIn("operating-margin shift", v.reverse_dcf_summary)
        self.assertTrue(any(s.startswith("Reverse DCF") for s in v.data_quality.warnings))  # gap > 5 pp

    def test_price_near_fair_value_has_no_gap_warning(self):
        # Fixed market cap keeps the WACC weights (and so the DCF) independent of the price being tested.
        market = make_market(share_price=235.0).model_copy(update={"market_cap": 3_500_000.0})
        fair = _value(self.audit, self.forecast, market).implied_fair_value_per_share
        v = _value(self.audit, self.forecast, market.model_copy(update={"share_price": fair}))
        self.assertEqual(v.reverse_dcf_status, "solved")
        self.assertAlmostEqual(v.market_implied_growth_pct, v.modeled_revenue_cagr_pct, delta=1.0)
        self.assertFalse(any(s.startswith("Reverse DCF") for s in v.data_quality.warnings))

    def test_no_price_means_no_reverse_dcf(self):
        v = _value(self.audit, self.forecast, MarketInputs())
        self.assertEqual(v.reverse_dcf_status, "price_unavailable")
        self.assertIsNone(v.market_implied_growth_pct)
        self.assertIsNone(v.market_implied_margin_shift_pp)
        self.assertEqual(v.modeled_revenue_cagr_pct, self.forecast.revenue_cagr_pct)
        self.assertIn("No share price", v.reverse_dcf_summary)

    def test_no_forecast_means_no_reverse_dcf(self):
        v = _value(self.audit, None, make_market())
        self.assertEqual(v.reverse_dcf_status, "forecast_unavailable")
        self.assertIsNone(v.market_implied_growth_pct)
        self.assertIsNone(v.modeled_revenue_cagr_pct)

    def test_unreachable_prices_return_none_with_warning(self):
        high = _value(self.audit, self.forecast, make_market(share_price=50_000.0))
        self.assertEqual((high.reverse_dcf_status, high.market_implied_growth_pct), ("price_above_range", None))
        self.assertTrue(any("exceeds the DCF value even at 60%" in w for w in high.data_quality.warnings))
        self.assertIsNone(high.market_implied_margin_shift_pp)  # +50 pp everywhere still falls short
        self.assertIn("No uniform operating-margin shift", high.reverse_dcf_summary)
        low = _value(self.audit, self.forecast, make_market(share_price=0.5))
        self.assertEqual((low.reverse_dcf_status, low.market_implied_growth_pct), ("price_below_range", None))

    def test_template_and_prompt_carry_reverse_dcf(self):
        out = make_valuation_output()
        self.assertIn("Reverse DCF:", out.valuation_summary)
        query = render_prompt("valuation_specialist", "prompt_valuation_query.j2", v=out)
        self.assertIn("- Reverse DCF:", query)
        self.assertIn("WACC rows x terminal RONIC columns", query)
        self.assertIn("EV/NOPAT", query)
        self.assertIn("WACC x RONIC", ValuationSpecialistAgent._template_summary(out))


class TestValuationRonicAndMultiples(unittest.TestCase):
    def test_steady_state_ronic_drives_terminal_value(self):
        forecast = make_forecast_output()
        self.assertEqual(forecast.diagnostics.steady_state_ronic_pct, 30.0)
        v = _value(make_aapl_audit(), forecast, make_market())
        self.assertEqual(v.ronic_used_pct, 30.0)
        self.assertEqual(v.terminal_method, "value_driver")
        self.assertIsNotNone(v.implied_terminal_ev_nopat)
        self.assertTrue(v.ronic_sensitivity_matrix_markdown.startswith("| WACC \\ RONIC"))

    def test_steady_state_below_wacc_is_floored_and_explained(self):
        forecast = make_forecast_output()
        diag = forecast.diagnostics.model_copy(update={"steady_state_ronic_pct": 8.0})
        v = _value(make_aapl_audit(), forecast.model_copy(update={"diagnostics": diag}), make_market())
        self.assertAlmostEqual(v.ronic_used_pct, v.wacc_audit.wacc * 100.0, places=2)
        self.assertTrue(any("more conservative than the terminal value" in w for w in v.data_quality.warnings))
        self.assertIn("creates no value", v.sensitivity_matrix_markdown)


class TestValuationCompanyProfiles(unittest.TestCase):
    """build_valuation must not crash and must flag problems for any company shape."""

    def _check(self, v):
        self.assertIn(v.reverse_dcf_status, REVERSE_STATUSES)
        if v.reverse_dcf_status != "solved":
            self.assertIsNone(v.market_implied_growth_pct)
        self.assertTrue(v.reverse_dcf_summary)

    def test_loss_maker(self):
        audit = make_loss_audit()
        v = _value(audit, make_forecast_output("LOSS", audit=audit), make_market(share_price=10.0, beta=1.5))
        self._check(v)
        self.assertEqual(v.terminal_method, "last_fcf_growth")
        self.assertIsNone(v.implied_terminal_ev_nopat)
        self.assertTrue(any("not positive" in w for w in v.data_quality.warnings))

    def test_net_debt_shrinking_company(self):
        audit = make_shrinking_audit()
        self.assertGreater(audit.balance_sheet.net_debt, 0)
        v = _value(audit, make_forecast_output("SHRK", audit=audit), make_market(share_price=5.0, beta=1.0))
        self._check(v)
        self.assertEqual(v.net_debt, audit.balance_sheet.net_debt)

    def test_heavy_capex_net_cash_company(self):
        audit = make_heavy_capex_audit()
        self.assertLess(audit.balance_sheet.net_debt, 0)
        forecast = make_forecast_output("HCX", audit=audit)
        self.assertEqual(forecast.forecast_horizon_years, 10)
        v = _value(audit, forecast, make_market(share_price=200.0, beta=1.3))
        self._check(v)
        self.assertEqual(v.reverse_dcf_horizon_years, 10)
        self.assertGreater(v.implied_fair_value_per_share, 0)

    def test_no_price_no_beta_profile(self):
        audit = make_heavy_capex_audit()
        v = _value(audit, make_forecast_output("HCX", audit=audit), MarketInputs())
        self._check(v)
        self.assertEqual(v.reverse_dcf_status, "price_unavailable")
        self.assertIsNone(v.valuation_stance)


if __name__ == "__main__":
    unittest.main()
