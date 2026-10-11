"""Tests for forecast assembly: fade extension, scenarios and diagnostics."""

import unittest

from app.agents.tools.forecast_baseline import build_forecast_baseline
from app.agents.tools.forecast_drivers import DEFAULT_TERMINAL_GROWTH, ForecastDrivers, historical_fade_drivers
from app.agents.tools.forecast_model import build_forecast
from tests.forecast_fixtures import make_aapl_audit, make_loss_audit


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

    def test_negative_net_reinvestment_is_flagged(self):
        drivers = self._drivers([0.05] * 5, capex_pct_of_revenue=0.01, da_pct_of_revenue=0.05)
        build = build_forecast(self.aapl, drivers, DEFAULT_TERMINAL_GROWTH)
        self.assertLess(build.diagnostics.final_year_reinvestment_rate_pct, 0)
        self.assertIsNone(build.diagnostics.ronic_pct)
        self.assertTrue(any("net reinvestment is negative" in w for w in build.diagnostics.warnings))

    def test_implied_growth_depends_on_reinvestment(self):
        low = build_forecast(self.aapl, self._drivers([0.05] * 5, capex_pct_of_revenue=0.04), DEFAULT_TERMINAL_GROWTH)
        high = build_forecast(self.aapl, self._drivers([0.05] * 5, capex_pct_of_revenue=0.10), DEFAULT_TERMINAL_GROWTH)
        self.assertIsNotNone(low.diagnostics.base_roic_pct)
        self.assertGreater(
            high.diagnostics.implied_fundamental_growth_pct, low.diagnostics.implied_fundamental_growth_pct
        )

    def test_ronic_suppressed_when_net_reinvestment_is_tiny(self):
        drivers = self._drivers([0.05] * 5, capex_pct_of_revenue=0.0276, da_pct_of_revenue=0.0275)
        self.assertIsNone(build_forecast(self.aapl, drivers, DEFAULT_TERMINAL_GROWTH).diagnostics.ronic_pct)

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


if __name__ == "__main__":
    unittest.main()
