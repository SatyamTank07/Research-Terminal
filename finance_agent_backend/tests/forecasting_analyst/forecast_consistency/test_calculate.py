"""Tests for the forecast cross-checks: reinvestment rate, RONIC, growth support and the warnings they raise."""

import unittest

from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.specialized.forecasting_analyst.modules.forecast_consistency import compute_diagnostics
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule import run_schedule
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path import apply_reinvestment_path
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import historical_fade_drivers
from tests.forecast_fixtures import make_aapl_audit


class TestComputeDiagnostics(unittest.TestCase):
    def setUp(self):
        self.baseline = build_forecast_baseline(make_aapl_audit())

    def _diagnostics(self, terminal_growth=DEFAULT_TERMINAL_GROWTH, **driver_updates):
        drivers = historical_fade_drivers(self.baseline, terminal_growth).model_copy(update=driver_updates)
        applied, _ = apply_reinvestment_path(self.baseline, drivers, terminal_growth)
        schedule = run_schedule(self.baseline, applied)
        return compute_diagnostics(self.baseline, applied, schedule, terminal_growth, False), schedule

    def test_final_year_reinvestment_rate_hits_the_g_over_ronic_target(self):
        diag, _ = self._diagnostics()
        self.assertAlmostEqual(diag.final_year_reinvestment_rate_pct, diag.target_reinvestment_rate_pct, delta=0.05)
        self.assertEqual(diag.terminal_growth_pct, round(DEFAULT_TERMINAL_GROWTH * 100.0, 2))

    def test_baseline_warnings_come_first(self):
        baseline = self.baseline.model_copy(update={"warnings": ["Baseline note."]})
        drivers, _ = apply_reinvestment_path(
            baseline, historical_fade_drivers(baseline, DEFAULT_TERMINAL_GROWTH), DEFAULT_TERMINAL_GROWTH
        )
        diag = compute_diagnostics(baseline, drivers, run_schedule(baseline, drivers), DEFAULT_TERMINAL_GROWTH, False)
        self.assertEqual(diag.warnings[0], "Baseline note.")

    def test_ronic_is_withheld_when_net_reinvestment_is_negligible(self):
        # Zero growth with CapEx at D&A: the steady state is maintenance level, so RONIC would be noise.
        diag, _ = self._diagnostics(
            terminal_growth=0.0, revenue_growth=[0.0] * 5, operating_margin=[0.32] * 5,
            capex_pct_of_revenue=0.0276, da_pct_of_revenue=0.0275, nwc_pct_of_incremental_revenue=0.0,
        )
        self.assertIsNone(diag.ronic_pct)

    def test_target_rate_is_withheld_under_a_maintenance_fade(self):
        diag, _ = self._diagnostics(operating_margin=[-0.05] * 5)
        self.assertIsNone(diag.target_reinvestment_rate_pct)
        self.assertIsNone(diag.final_year_reinvestment_rate_pct)


if __name__ == "__main__":
    unittest.main()
