"""Tests for forecast driver fades and LLM proposal screening."""

import unittest

from app.agents.tools.forecast_baseline import build_forecast_baseline
from app.agents.tools.forecast_drivers import (
    BASE_HORIZON_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    EXTENDED_HORIZON_YEARS,
    ForecastDriverProposal,
    extend_fade,
    historical_fade_drivers,
    linear_fade,
    screen_forecast_drivers,
)
from tests.forecast_fixtures import make_aapl_audit

ALLOWED = {"c1", "c2"}


def _proposal(**overrides) -> ForecastDriverProposal:
    base = dict(
        revenue_growth_pct=[7.0, 6.5, 6.0, 5.0, 4.0],
        operating_margin_pct=[32.0, 32.5, 33.0, 33.0, 33.0],
    )
    return ForecastDriverProposal(**{**base, **overrides})


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


class TestScreenForecastDrivers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aapl = build_forecast_baseline(make_aapl_audit())

    def test_accepts_in_bounds_proposal_and_converts_to_decimals(self):
        r = screen_forecast_drivers(_proposal(), self.aapl, ALLOWED)
        self.assertTrue(r.accepted)
        self.assertEqual(r.drivers.revenue_growth, [0.07, 0.065, 0.06, 0.05, 0.04])
        self.assertEqual(r.drivers.operating_margin, [0.32, 0.325, 0.33, 0.33, 0.33])
        self.assertEqual(r.adjustments, [])
        self.assertAlmostEqual(r.drivers.tax_rate, self.aapl.tax_rate)

    def test_rejects_wrong_length(self):
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[7.0, 6.0, 5.0, 4.0]), self.aapl, ALLOWED)
        self.assertFalse(r.accepted)
        self.assertIn("exactly 5", r.errors[0])

    def test_rejects_decimal_margins(self):
        r = screen_forecast_drivers(_proposal(operating_margin_pct=[0.32] * 5), self.aapl, ALLOWED)
        self.assertFalse(r.accepted)
        self.assertIn("WHOLE percent", r.errors[0])

    def test_clamps_year1_growth_to_band(self):
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[40.0, 8.0, 7.0, 6.0, 5.0]), self.aapl, ALLOWED)
        self.assertEqual(r.drivers.revenue_growth[0], 0.2143)
        adj = r.adjustments[0]
        self.assertEqual((adj.driver, adj.year, adj.action), ("revenue_growth", 1, "clamped"))
        self.assertEqual(adj.proposed, 0.4)

    def test_clamps_margin_steps(self):
        r = screen_forecast_drivers(_proposal(operating_margin_pct=[40.0, 40.0, 33.0, 33.0, 33.0]), self.aapl, ALLOWED)
        self.assertEqual(r.drivers.operating_margin, [0.3497, 0.3797, 0.3497, 0.33, 0.33])
        self.assertEqual(sum(a.driver == "operating_margin" for a in r.adjustments), 3)

    def test_hypergrowth_year1_band_widens(self):
        hyper = self.aapl.model_copy(update={"latest_revenue_growth": 1.14})
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[70.0, 50.0, 35.0, 25.0, 15.0]), hyper, ALLOWED)
        self.assertEqual(r.drivers.revenue_growth[0], 0.70)
        self.assertEqual(r.adjustments, [])
        capped = screen_forecast_drivers(_proposal(revenue_growth_pct=[150.0, 50.0, 35.0, 25.0, 15.0]), hyper, ALLOWED)
        self.assertEqual(capped.drivers.revenue_growth[0], 1.0)

    def test_year1_deceleration_after_spike_is_not_forced_up(self):
        # Acquisition-driven +44% year followed by a normal year: 7% must be accepted as-is.
        spike = self.aapl.model_copy(update={"latest_revenue_growth": 0.44})
        r = screen_forecast_drivers(_proposal(), spike, ALLOWED)
        self.assertEqual(r.drivers.revenue_growth[0], 0.07)
        self.assertEqual(r.adjustments, [])

    def test_year1_recovery_after_collapse_is_not_forced_down(self):
        collapse = self.aapl.model_copy(update={"latest_revenue_growth": -0.64})
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[-10.0, 0.0, 3.0, 3.0, 3.0]), collapse, ALLOWED)
        self.assertEqual(r.drivers.revenue_growth[0], -0.10)
        deeper = screen_forecast_drivers(_proposal(revenue_growth_pct=[-50.0, 0.0, 3.0, 3.0, 3.0]), collapse, ALLOWED)
        self.assertEqual(deeper.drivers.revenue_growth[0], -0.30)

    def test_low_margin_company_whole_percent_margins_accepted(self):
        thin = self.aapl.model_copy(update={"base_operating_margin": 0.011})
        r = screen_forecast_drivers(_proposal(operating_margin_pct=[0.8, 0.85, 0.9, 0.95, 0.95]), thin, ALLOWED)
        self.assertTrue(r.accepted, r.errors)
        self.assertEqual(r.drivers.operating_margin[0], 0.008)

    def test_breakeven_company_margins_accepted(self):
        breakeven = self.aapl.model_copy(update={"base_operating_margin": -0.015})
        r = screen_forecast_drivers(_proposal(operating_margin_pct=[-0.8, -0.4, 0.0, 0.4, 0.8]), breakeven, ALLOWED)
        self.assertTrue(r.accepted, r.errors)

    def test_rejects_decimal_growth(self):
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[0.07, 0.065, 0.06, 0.05, 0.04]), self.aapl, ALLOWED)
        self.assertFalse(r.accepted)
        self.assertIn("WHOLE percent", r.errors[0])

    def test_rejects_decimal_growth_far_from_anchor(self):
        hyper = self.aapl.model_copy(update={"latest_revenue_growth": 1.14})
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[0.6, 0.45, 0.3, 0.2, 0.1]), hyper, ALLOWED)
        self.assertFalse(r.accepted)
        self.assertIn("WHOLE percent", r.errors[0])
        declining = self.aapl.model_copy(update={"latest_revenue_growth": -0.10})
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[-0.05, -0.02, 0.0, 0.01, 0.02]), declining, ALLOWED)
        self.assertFalse(r.accepted)

    def test_whole_percent_decline_near_zero_accepted(self):
        declining = self.aapl.model_copy(update={"latest_revenue_growth": -0.10})
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[-0.5, 0.0, 0.5, 0.8, 0.9]), declining, ALLOWED)
        self.assertTrue(r.accepted, r.errors)

    def test_low_growth_whole_percent_accepted(self):
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[0.5, 0.5, 0.8, 1.0, 1.0]), self.aapl, ALLOWED)
        self.assertTrue(r.accepted, r.errors)

    def test_capex_override_needs_citation(self):
        r = screen_forecast_drivers(_proposal(capex_pct_of_revenue=3.5), self.aapl, ALLOWED)
        self.assertAlmostEqual(r.drivers.capex_pct_of_revenue, self.aapl.capex_pct_of_revenue)
        self.assertEqual(r.adjustments[0].action, "rejected")

        cited = screen_forecast_drivers(
            _proposal(capex_pct_of_revenue=3.5, capex_citation_chunk_id="c1"), self.aapl, ALLOWED
        )
        self.assertEqual(cited.drivers.capex_pct_of_revenue, 0.035)
        self.assertEqual(cited.adjustments, [])

    def test_capex_override_clamped_to_band(self):
        r = screen_forecast_drivers(
            _proposal(capex_pct_of_revenue=10.0, capex_citation_chunk_id="c1"), self.aapl, ALLOWED
        )
        self.assertEqual(r.drivers.capex_pct_of_revenue, 0.0694)
        self.assertEqual(r.adjustments[0].action, "clamped")

    def test_nwc_override_with_citation(self):
        r = screen_forecast_drivers(
            _proposal(nwc_pct_of_incremental_revenue=-5.0, nwc_citation_chunk_id="c2"), self.aapl, ALLOWED
        )
        self.assertEqual(r.drivers.nwc_pct_of_incremental_revenue, -0.05)

    def test_evidence_ids_filtered_to_known_chunks(self):
        r = screen_forecast_drivers(_proposal(evidence_chunk_ids=["c1", "bogus"]), self.aapl, ALLOWED)
        self.assertEqual(r.evidence_chunk_ids, ["c1"])


if __name__ == "__main__":
    unittest.main()
