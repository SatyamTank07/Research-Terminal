"""Tests for the reinvestment path: CapEx fades to the steady state where net reinvestment = g / RONIC x NOPAT."""

import unittest

from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path import apply_reinvestment_path
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import historical_fade_drivers
from tests.forecast_fixtures import make_aapl_audit


class TestApplyReinvestmentPath(unittest.TestCase):
    def setUp(self):
        self.baseline = build_forecast_baseline(make_aapl_audit())
        self.drivers = historical_fade_drivers(self.baseline, DEFAULT_TERMINAL_GROWTH)

    def test_final_year_net_reinvestment_funds_terminal_growth(self):
        applied, _ = apply_reinvestment_path(self.baseline, self.drivers, DEFAULT_TERMINAL_GROWTH)
        g_n = applied.revenue_growth[-1]
        nopat_margin = applied.operating_margin[-1] * (1.0 - applied.tax_rate)
        net_reinvestment = (
            applied.steady_state_capex_pct_of_revenue - applied.da_pct_of_revenue
            + applied.nwc_pct_of_incremental_revenue * g_n / (1.0 + g_n)
        )
        target = DEFAULT_TERMINAL_GROWTH / self.baseline.steady_state_ronic * nopat_margin
        self.assertEqual(applied.reinvestment_method, "value_driver_fade")
        self.assertAlmostEqual(net_reinvestment, target, places=4)

    def test_capex_fades_linearly_from_the_year_one_intensity(self):
        applied, _ = apply_reinvestment_path(self.baseline, self.drivers, DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(len(applied.capex_path), len(applied.revenue_growth))
        self.assertAlmostEqual(applied.capex_path[0], round(self.drivers.capex_pct_of_revenue, 4))
        self.assertEqual(applied.capex_path[-1], applied.steady_state_capex_pct_of_revenue)

    def test_unprofitable_final_year_fades_to_maintenance_with_a_warning(self):
        losing = self.drivers.model_copy(update={"operating_margin": [-0.05] * len(self.drivers.operating_margin)})
        applied, warnings = apply_reinvestment_path(self.baseline, losing, DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(applied.reinvestment_method, "maintenance_fade")
        self.assertEqual(applied.steady_state_capex_pct_of_revenue, round(losing.da_pct_of_revenue, 4))
        self.assertIn("maintenance", warnings[0])

    def test_a_large_capex_transition_is_warned(self):
        heavy = self.drivers.model_copy(update={"capex_pct_of_revenue": 0.40})
        _, warnings = apply_reinvestment_path(self.baseline, heavy, DEFAULT_TERMINAL_GROWTH)
        self.assertTrue(any("large reinvestment transition" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
