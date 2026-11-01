"""Tests for the bull / base / bear scenarios: weights, ordering, risk scaling and the cash-burner flip."""

import unittest

from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule import run_schedule
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path import apply_reinvestment_path
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import historical_fade_drivers
from app.agents.specialized.forecasting_analyst.modules.scenarios import SCENARIO_PROBABILITIES, build_scenarios
from tests.forecast_fixtures import make_aapl_audit


class TestBuildScenarios(unittest.TestCase):
    def setUp(self):
        self.baseline = build_forecast_baseline(make_aapl_audit())
        drivers = historical_fade_drivers(self.baseline, DEFAULT_TERMINAL_GROWTH)
        self.drivers, _ = apply_reinvestment_path(self.baseline, drivers, DEFAULT_TERMINAL_GROWTH)
        self.schedule = run_schedule(self.baseline, self.drivers)

    def _scenarios(self, risk_profile=None, schedule=None):
        return build_scenarios(
            self.baseline, self.drivers, schedule or self.schedule, risk_profile, DEFAULT_TERMINAL_GROWTH
        )

    def test_cases_come_bull_base_bear_with_25_50_25_weights(self):
        scenarios = self._scenarios()
        self.assertEqual([s.name for s in scenarios], ["bull", "base", "bear"])
        self.assertEqual([s.probability for s in scenarios], [SCENARIO_PROBABILITIES[n] for n in ("bull", "base", "bear")])
        self.assertEqual(scenarios[1].projected_fcfs, self.schedule["projected_fcfs"])

    def test_bull_beats_base_beats_bear_for_a_cash_generator(self):
        bull, base, bear = self._scenarios()
        self.assertGreater(bull.cumulative_fcf, base.cumulative_fcf)
        self.assertGreater(base.cumulative_fcf, bear.cumulative_fcf)

    def test_a_high_risk_profile_deepens_only_the_bear_case(self):
        calm, risky = self._scenarios("Low"), self._scenarios("High")
        self.assertEqual(calm[0].cumulative_fcf, risky[0].cumulative_fcf)
        self.assertLess(risky[2].cumulative_fcf, calm[2].cumulative_fcf)

    def test_growth_shifts_vanish_by_the_final_year(self):
        bull, base, bear = self._scenarios()
        self.assertEqual(bull.revenue_growth_pct[-1], base.revenue_growth_pct[-1])
        self.assertEqual(bear.revenue_growth_pct[-1], base.revenue_growth_pct[-1])


if __name__ == "__main__":
    unittest.main()
