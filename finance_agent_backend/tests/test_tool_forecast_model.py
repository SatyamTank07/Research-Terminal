"""Tests for forecast assembly: fade extension, scenarios and diagnostics."""

import unittest

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import ForecastDrivers, historical_fade_drivers
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule import project_with_overrides
from app.agents.tools.forecast_model import build_forecast
from tests.forecast_fixtures import (
    AAPL_BALANCE_SHEET,
    AAPL_ROWS,
    make_aapl_audit,
    make_audit,
    make_heavy_capex_audit,
    make_loss_audit,
    make_shrinking_audit,
)


class TestForecastModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aapl = build_forecast_baseline(make_aapl_audit())

    def _drivers(self, growth, margins=None, **overrides) -> ForecastDrivers:
        base = dict(
            revenue_growth=growth, operating_margin=margins or [0.32] * len(growth),
            tax_rate=self.aapl.tax_rate, capex_pct_of_revenue=self.aapl.capex_pct_of_revenue,
            da_pct_of_revenue=self.aapl.da_pct_of_revenue, nwc_pct_of_incremental_revenue=0.0,
            sbc_pct_of_revenue=0.0,
        )
        return ForecastDrivers(**{**base, **overrides})

    def test_fallback_forecast_scenarios_bracket_the_base_case(self):
        build = build_forecast(self.aapl, historical_fade_drivers(self.aapl, DEFAULT_TERMINAL_GROWTH), DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(build.schedule["forecast_horizon_years"], 5)
        self.assertEqual([s.name for s in build.scenarios], ["bull", "base", "bear"])
        self.assertAlmostEqual(sum(s.probability for s in build.scenarios), 1.0)
        bull, base, bear = build.scenarios
        self.assertGreater(bull.cumulative_fcf, base.cumulative_fcf)
        self.assertGreater(base.cumulative_fcf, bear.cumulative_fcf)
        self.assertEqual(base.projected_fcfs, build.schedule["projected_fcfs"])
        # all scenarios converge to the same terminal growth
        self.assertEqual({s.revenue_growth_pct[-1] for s in build.scenarios}, {2.5})
        self.assertFalse(build.diagnostics.fade_extended)
        self.assertEqual(build.diagnostics.final_year_growth_pct, 2.5)
        self.assertEqual(len(build.preview_rows()), 5)

    def test_far_from_terminal_growth_extends_to_ten_years(self):
        build = build_forecast(self.aapl, self._drivers([0.30, 0.25, 0.20, 0.18, 0.15]), DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(build.schedule["forecast_horizon_years"], 10)
        self.assertTrue(build.diagnostics.fade_extended)
        self.assertEqual(build.drivers.revenue_growth[-1], DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(len(build.scenarios[0].projected_fcfs), 10)

    def test_high_risk_profile_deepens_bear_case(self):
        drivers = historical_fade_drivers(self.aapl, DEFAULT_TERMINAL_GROWTH)
        calm = build_forecast(self.aapl, drivers, DEFAULT_TERMINAL_GROWTH, risk_profile="Low")
        risky = build_forecast(self.aapl, drivers, DEFAULT_TERMINAL_GROWTH, risk_profile="High")
        self.assertLess(risky.scenarios[2].cumulative_fcf, calm.scenarios[2].cumulative_fcf)
        self.assertEqual(risky.scenarios[1].projected_fcfs, calm.scenarios[1].projected_fcfs)

    def test_asset_light_capex_below_da_fades_up_to_fund_terminal_growth(self):
        # CapEx 1% vs D&A 5%: year 1 shrinks the asset base, the final year funds 2.5% growth at a 30% RONIC.
        drivers = self._drivers([0.05, 0.045, 0.04, 0.03, 0.025], capex_pct_of_revenue=0.01, da_pct_of_revenue=0.05)
        build = build_forecast(self.aapl, drivers, DEFAULT_TERMINAL_GROWTH)
        d = build.drivers
        self.assertEqual(d.reinvestment_method, "value_driver_fade")
        self.assertEqual(d.capex_path[0], 0.01)
        nopat_margin = 0.32 * (1 - self.aapl.tax_rate)
        self.assertAlmostEqual(d.steady_state_capex_pct_of_revenue, 0.05 + 0.025 / 0.30 * nopat_margin, places=4)
        self.assertEqual(d.capex_path[-1], d.steady_state_capex_pct_of_revenue)
        self.assertAlmostEqual(build.diagnostics.final_year_reinvestment_rate_pct, 2.5 / 0.30, delta=0.05)
        self.assertEqual(build.diagnostics.target_reinvestment_rate_pct, round(2.5 / 0.30, 2))
        self.assertEqual(build.diagnostics.steady_state_ronic_pct, 30.0)
        self.assertFalse(any("net reinvestment is negative" in w for w in build.diagnostics.warnings))

    def test_terminal_reinvestment_does_not_depend_on_starting_capex(self):
        low = build_forecast(self.aapl, self._drivers([0.05] * 5, capex_pct_of_revenue=0.04), DEFAULT_TERMINAL_GROWTH)
        high = build_forecast(self.aapl, self._drivers([0.05] * 5, capex_pct_of_revenue=0.10), DEFAULT_TERMINAL_GROWTH)
        self.assertIsNotNone(low.diagnostics.base_roic_pct)
        self.assertEqual(low.drivers.capex_path[-1], high.drivers.capex_path[-1])
        self.assertEqual(
            low.diagnostics.implied_fundamental_growth_pct, high.diagnostics.implied_fundamental_growth_pct
        )
        self.assertGreater(low.schedule["cumulative_fcf"], high.schedule["cumulative_fcf"])

    def test_ronic_suppressed_when_net_reinvestment_is_tiny(self):
        # Zero growth, CapEx at D&A: the steady state is maintenance-level and net reinvestment ~0.
        drivers = self._drivers([0.0] * 5, capex_pct_of_revenue=0.0276, da_pct_of_revenue=0.0275)
        build = build_forecast(self.aapl, drivers, 0.0)
        self.assertEqual(build.drivers.steady_state_capex_pct_of_revenue, 0.0275)
        self.assertIsNone(build.diagnostics.ronic_pct)

    def test_cash_burning_company_bull_case_burns_less(self):
        burner = build_forecast_baseline(make_loss_audit()).model_copy(update={"base_operating_margin": -0.40})
        build = build_forecast(burner, historical_fade_drivers(burner, DEFAULT_TERMINAL_GROWTH), DEFAULT_TERMINAL_GROWTH)
        bull, base, bear = build.scenarios
        self.assertGreater(bull.cumulative_fcf, base.cumulative_fcf)
        self.assertGreater(base.cumulative_fcf, bear.cumulative_fcf)

    def test_loss_making_baseline_builds(self):
        loss = build_forecast_baseline(make_loss_audit())
        build = build_forecast(loss, historical_fade_drivers(loss, DEFAULT_TERMINAL_GROWTH), DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(build.schedule["forecast_horizon_years"], 10)
        self.assertTrue(all(f < 0 for f in build.schedule["projected_fcfs"]))
        self.assertIsNone(build.diagnostics.final_year_reinvestment_rate_pct)
        self.assertIsNone(build.diagnostics.ronic_pct)
        self.assertTrue(any("D&A not reported" in w for w in build.diagnostics.warnings))


class TestReinvestmentEdgeProfiles(unittest.TestCase):
    """The CapEx steady state must stay sane for any company shape."""

    def _build(self, baseline, terminal_growth=DEFAULT_TERMINAL_GROWTH):
        return build_forecast(baseline, historical_fade_drivers(baseline, terminal_growth), terminal_growth)

    def _assert_engine_safe(self, build):
        n = len(build.drivers.revenue_growth)
        self.assertEqual(len(build.drivers.capex_path), n)
        self.assertTrue(all(0.0 <= c <= 1.0 for c in build.drivers.capex_path))
        for s in build.scenarios:
            self.assertEqual(len(s.projected_fcfs), n)

    def test_heavy_capex_step_up_fades_to_steady_state(self):
        b = build_forecast_baseline(make_heavy_capex_audit())
        self.assertAlmostEqual(b.latest_capex_pct_of_revenue, 0.18)
        self.assertAlmostEqual(b.capex_pct_of_revenue, (0.09 + 2 * 0.11 + 3 * 0.18) / 6, places=6)
        build = self._build(b)
        self._assert_engine_safe(build)
        d = build.drivers
        self.assertEqual(d.capex_path[0], round(b.capex_pct_of_revenue, 4))
        self.assertTrue(all(a >= c for a, c in zip(d.capex_path, d.capex_path[1:])))  # monotone fade down
        self.assertLess(d.steady_state_capex_pct_of_revenue, 0.12)
        self.assertGreater(d.steady_state_capex_pct_of_revenue, d.da_pct_of_revenue)
        # Final-year net reinvestment equals g / RONIC_ss of NOPAT, so the forecast RONIC is not far below ROIC.
        target = DEFAULT_TERMINAL_GROWTH / b.steady_state_ronic * 100
        self.assertAlmostEqual(build.diagnostics.final_year_reinvestment_rate_pct, target, delta=0.1)
        self.assertEqual(build.diagnostics.target_reinvestment_rate_pct, round(target, 2))

    def test_extreme_capex_transition_is_warned(self):
        b = build_forecast_baseline(make_heavy_capex_audit()).model_copy(update={"capex_pct_of_revenue": 0.40})
        build = self._build(b)
        self.assertTrue(any("large reinvestment transition" in w for w in build.diagnostics.warnings))

    def test_loss_maker_fades_to_maintenance(self):
        loss = build_forecast_baseline(make_loss_audit())
        build = self._build(loss)
        self._assert_engine_safe(build)
        self.assertEqual(build.drivers.reinvestment_method, "maintenance_fade")
        self.assertEqual(build.drivers.steady_state_capex_pct_of_revenue, round(build.drivers.da_pct_of_revenue, 4))
        self.assertIsNone(build.diagnostics.target_reinvestment_rate_pct)
        self.assertTrue(any("NOPAT is not positive" in w for w in build.diagnostics.warnings))

    def test_negative_terminal_nopat_after_profitable_base(self):
        b = build_forecast_baseline(make_aapl_audit())
        drivers = historical_fade_drivers(b, DEFAULT_TERMINAL_GROWTH).model_copy(
            update={"operating_margin": [0.20, 0.10, 0.0, -0.05, -0.10]}
        )
        build = build_forecast(b, drivers, DEFAULT_TERMINAL_GROWTH)
        self._assert_engine_safe(build)
        self.assertEqual(build.drivers.reinvestment_method, "maintenance_fade")
        self.assertIsNone(build.diagnostics.final_year_reinvestment_rate_pct)

    def test_zero_or_negative_roic_uses_default_ronic(self):
        b = build_forecast_baseline(make_aapl_audit()).model_copy(
            update={"base_roic": None, "steady_state_ronic": 0.10, "steady_state_ronic_source": "default_no_positive_roic"}
        )
        build = self._build(b)
        self.assertIsNone(build.diagnostics.base_roic_pct)
        self.assertIsNone(build.diagnostics.implied_fundamental_growth_pct)
        self.assertEqual(build.diagnostics.steady_state_ronic_pct, 10.0)

    def test_missing_da_and_zero_capex(self):
        rows = [dict(r, capital_expenditures=0.0) for r in AAPL_ROWS]
        rows[-1].pop("depreciation_amortization")
        b = build_forecast_baseline(make_audit("NOCAPEX", rows, AAPL_BALANCE_SHEET))
        self.assertEqual((b.capex_pct_of_revenue, b.da_pct_of_revenue, b.da_source), (0.0, 0.0, "capex_parity_proxy"))
        build = self._build(b)
        self._assert_engine_safe(build)
        self.assertEqual(build.drivers.capex_path[0], 0.0)
        self.assertGreater(build.drivers.capex_path[-1], 0.0)  # growth still has to be funded

    def test_hypergrowth_extends_and_fades(self):
        b = build_forecast_baseline(make_aapl_audit()).model_copy(update={"historical_cagr": 0.90})
        build = self._build(b)
        self._assert_engine_safe(build)
        self.assertEqual(len(build.drivers.revenue_growth), 10)
        self.assertEqual(build.drivers.revenue_growth[0], 0.40)

    def test_shrinking_revenue_and_negative_terminal_growth(self):
        b = build_forecast_baseline(make_shrinking_audit())
        self.assertLess(b.historical_cagr, 0)
        self.assertEqual(b.margin_trend, "contracting")
        build = self._build(b, terminal_growth=-0.01)
        self._assert_engine_safe(build)
        d = build.drivers
        self.assertEqual(d.reinvestment_method, "value_driver_fade")
        self.assertLess(d.steady_state_capex_pct_of_revenue, d.da_pct_of_revenue)  # the asset base shrinks

    def test_large_positive_working_capital_floors_capex_at_zero(self):
        b = build_forecast_baseline(make_aapl_audit())
        drivers = self._drivers_for(b, growth=[0.30] * 5, nwc=0.5, da=0.0)
        build = build_forecast(b, drivers, DEFAULT_TERMINAL_GROWTH)
        self._assert_engine_safe(build)
        self.assertEqual(build.drivers.steady_state_capex_pct_of_revenue, 0.0)
        self.assertTrue(any("Steady-state CapEx" in w for w in build.diagnostics.warnings))

    @staticmethod
    def _drivers_for(b, growth, nwc, da):
        return historical_fade_drivers(b, DEFAULT_TERMINAL_GROWTH).model_copy(update={
            "revenue_growth": growth, "operating_margin": [0.05] * len(growth),
            "nwc_pct_of_incremental_revenue": nwc, "da_pct_of_revenue": da,
        })

    def test_project_with_overrides_matches_build_for_same_path(self):
        b = build_forecast_baseline(make_heavy_capex_audit())
        build = self._build(b)
        fcfs, nopat = project_with_overrides(b, build.drivers, DEFAULT_TERMINAL_GROWTH, growth=build.drivers.revenue_growth)
        self.assertEqual(fcfs, build.schedule["projected_fcfs"])
        self.assertEqual(nopat, build.schedule["forecast_schedule"][-1]["projected_nopat"])


if __name__ == "__main__":
    unittest.main()
