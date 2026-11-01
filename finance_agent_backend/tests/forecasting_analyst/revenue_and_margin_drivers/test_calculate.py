"""Tests for the code-built growth paths: linear fade, historical fade fallback and the 10-year fade extension."""

import unittest

from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import extend_fade, historical_fade_drivers, linear_fade
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers.thresholds import BASE_HORIZON_YEARS, EXTENDED_HORIZON_YEARS
from tests.forecast_fixtures import make_aapl_audit


class TestFades(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aapl = build_forecast_baseline(make_aapl_audit())

    def test_linear_fade_endpoints(self):
        self.assertEqual(linear_fade(0.10, 0.02, 5), [0.1, 0.08, 0.06, 0.04, 0.02])
        self.assertEqual(linear_fade(0.10, 0.02, 1), [0.02])

    def test_fallback_fades_aapl_to_terminal_in_five_years(self):
        d = historical_fade_drivers(self.aapl, DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(len(d.revenue_growth), BASE_HORIZON_YEARS)
        self.assertAlmostEqual(d.revenue_growth[0], 0.0420, places=4)
        self.assertEqual(d.revenue_growth[-1], DEFAULT_TERMINAL_GROWTH)
        self.assertTrue(all(a > b for a, b in zip(d.revenue_growth, d.revenue_growth[1:])))
        self.assertEqual(d.operating_margin, [0.3197] * BASE_HORIZON_YEARS)
        self.assertAlmostEqual(d.capex_pct_of_revenue, self.aapl.capex_pct_of_revenue)

    def test_fallback_declining_company_reverts_upward(self):
        declining = self.aapl.model_copy(update={"historical_cagr": -0.10})
        d = historical_fade_drivers(declining, DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(len(d.revenue_growth), EXTENDED_HORIZON_YEARS)
        self.assertEqual(d.revenue_growth[0], -0.10)
        self.assertEqual(d.revenue_growth[-1], DEFAULT_TERMINAL_GROWTH)
        self.assertTrue(all(a < b for a, b in zip(d.revenue_growth, d.revenue_growth[1:])))

    def test_fallback_hypergrowth_is_capped_and_extended(self):
        d = historical_fade_drivers(self.aapl.model_copy(update={"historical_cagr": 0.80}), DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(d.revenue_growth[0], 0.40)
        self.assertEqual(len(d.revenue_growth), EXTENDED_HORIZON_YEARS)

    def test_single_year_baseline_flat_at_terminal(self):
        single = build_forecast_baseline(make_aapl_audit(years=(2025,)))
        d = historical_fade_drivers(single, DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(d.revenue_growth, [DEFAULT_TERMINAL_GROWTH] * BASE_HORIZON_YEARS)

    def test_extend_fade_adds_years_when_far_from_terminal(self):
        growth, margins, extended = extend_fade([0.30, 0.25, 0.20, 0.18, 0.15], [0.2] * 5, 0.025)
        self.assertTrue(extended)
        self.assertEqual(growth[5:], [0.125, 0.1, 0.075, 0.05, 0.025])
        self.assertEqual(margins, [0.2] * 10)

    def test_extend_fade_noop_when_close(self):
        growth, margins, extended = extend_fade([0.07, 0.06, 0.05, 0.045, 0.04], [0.2] * 5, 0.025)
        self.assertFalse(extended)
        self.assertEqual(len(growth), 5)


if __name__ == "__main__":
    unittest.main()
