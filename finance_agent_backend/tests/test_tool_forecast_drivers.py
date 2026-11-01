"""Tests for forecast driver fades and LLM proposal screening."""

import unittest

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.tools.forecast_drivers import (
    BASE_HORIZON_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    EXTENDED_HORIZON_YEARS,
    ForecastDriverProposal,
    extend_fade,
    historical_fade_drivers,
    linear_fade,
    screen_forecast_drivers,
    unsupported_margin_corridor,
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
        r = screen_forecast_drivers(
            _proposal(operating_margin_pct=[40.0, 40.0, 33.0, 33.0, 33.0], margin_citation_chunk_id="c1"),
            self.aapl, ALLOWED, margin_evidence_chunk_ids={"c1"},
        )
        self.assertTrue(r.margin_evidence_accepted)
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
        self.assertEqual(r.drivers.capex_pct_of_revenue, round(self.aapl.capex_pct_of_revenue * 2.5, 4))
        self.assertEqual(r.adjustments[0].action, "clamped")

    def test_nwc_override_with_citation(self):
        r = screen_forecast_drivers(
            _proposal(nwc_pct_of_incremental_revenue=-5.0, nwc_citation_chunk_id="c2"), self.aapl, ALLOWED
        )
        self.assertEqual(r.drivers.nwc_pct_of_incremental_revenue, -0.05)

    def test_evidence_ids_filtered_to_known_chunks(self):
        r = screen_forecast_drivers(_proposal(evidence_chunk_ids=["c1", "bogus"]), self.aapl, ALLOWED)
        self.assertEqual(r.evidence_chunk_ids, ["c1"])

    def test_non_finite_and_wrong_length_inputs_are_rejected_not_raised(self):
        r = screen_forecast_drivers(_proposal(operating_margin_pct=[30.0, float("nan"), 30.0, 30.0, 30.0]),
                                    self.aapl, ALLOWED)
        self.assertFalse(r.accepted)
        self.assertIn("non-finite", r.errors[0])
        r = screen_forecast_drivers(_proposal(revenue_growth_pct=[], operating_margin_pct=[]), self.aapl, ALLOWED)
        self.assertFalse(r.accepted)
        self.assertEqual(len(r.errors), 2)


class TestMarginEvidenceScreen(unittest.TestCase):
    """Unsupported margin paths stay in a corridor set by the audited trend; cited evidence lifts it."""

    @classmethod
    def setUpClass(cls):
        base = build_forecast_baseline(make_aapl_audit())
        # Expanding history (5.3% -> 11.16%), like a company that just widened margins for three years.
        cls.expanding = base.model_copy(update={
            "base_operating_margin": 0.1116, "margin_trend": "expanding", "historical_average_margin": 0.072,
        })
        cls.contracting = base.model_copy(update={
            "base_operating_margin": 0.0459, "margin_trend": "contracting", "historical_average_margin": 0.0701,
        })
        cls.mixed = base.model_copy(update={
            "base_operating_margin": 0.6038, "margin_trend": "mixed", "historical_average_margin": 0.5897,
        })

    def _screen(self, baseline, margins, citation=None, evidence=frozenset()):
        return screen_forecast_drivers(
            _proposal(operating_margin_pct=margins, margin_citation_chunk_id=citation),
            baseline, ALLOWED, margin_evidence_chunk_ids=evidence,
        )

    def test_unsupported_reversal_of_expanding_trend_is_held_at_base(self):
        r = self._screen(self.expanding, [11.0, 11.0, 11.0, 10.0, 9.0])
        self.assertTrue(r.accepted)
        self.assertEqual(r.drivers.operating_margin, [0.11, 0.11, 0.11, 0.1016, 0.1016])
        clamps = [a for a in r.adjustments if a.driver == "operating_margin"]
        self.assertEqual([a.year for a in clamps], [4, 5])
        self.assertIn("expanding audited margin trend", clamps[0].reason)
        self.assertIn("no margin_citation_chunk_id", clamps[0].reason)
        self.assertFalse(r.margin_evidence_accepted)

    def test_cited_margin_evidence_allows_the_reversal(self):
        r = self._screen(self.expanding, [11.0, 11.0, 11.0, 10.0, 9.0], citation="c1", evidence={"c1"})
        self.assertTrue(r.margin_evidence_accepted)
        self.assertEqual(r.drivers.operating_margin, [0.11, 0.11, 0.11, 0.10, 0.09])
        self.assertEqual(r.adjustments, [])

    def test_citation_without_margin_language_does_not_count(self):
        r = self._screen(self.expanding, [11.0, 11.0, 11.0, 10.0, 9.0], citation="c2", evidence={"c1"})
        self.assertFalse(r.margin_evidence_accepted)
        self.assertIn("citation 'c2' is not an MD&A chunk", r.adjustments[0].reason)
        unknown = self._screen(self.expanding, [9.0] * 5, citation="bogus", evidence={"bogus"})
        self.assertFalse(unknown.margin_evidence_accepted)  # not among the excerpts the agent saw

    def test_trend_consistent_expansion_is_allowed_up_to_allowance(self):
        r = self._screen(self.expanding, [12.0, 13.0, 14.0, 15.0, 16.0])
        self.assertEqual(r.drivers.operating_margin, [0.12, 0.13, 0.14, 0.1416, 0.1416])

    def test_contracting_trend_blocks_unsupported_recovery_but_allows_decline(self):
        recovery = self._screen(self.contracting, [5.0, 6.0, 7.0, 8.0, 9.0])
        self.assertEqual(max(recovery.drivers.operating_margin), 0.0559)
        decline = self._screen(self.contracting, [4.0, 3.5, 3.0, 2.0, 1.0])
        self.assertEqual(decline.drivers.operating_margin, [0.04, 0.035, 0.03, 0.02, 0.0159])

    def test_mixed_trend_allows_mean_reversion_only(self):
        r = self._screen(self.mixed, [60.0, 59.5, 59.0, 59.0, 59.0])
        self.assertEqual(r.adjustments, [])
        far = self._screen(self.mixed, [58.0, 55.0, 52.0, 50.0, 50.0])
        self.assertEqual(min(far.drivers.operating_margin), 0.5797)
        up = self._screen(self.mixed, [62.0, 64.0, 66.0, 66.0, 66.0])
        self.assertEqual(max(up.drivers.operating_margin), 0.6138)

    def test_single_year_history_is_noise_band_only(self):
        single = build_forecast_baseline(make_aapl_audit(years=(2025,)))
        lo, hi, _ = unsupported_margin_corridor(single)
        self.assertAlmostEqual(hi - lo, 0.02)
        r = self._screen(single, [40.0] * 5)
        self.assertEqual(r.drivers.operating_margin[-1], round(single.base_operating_margin + 0.01, 4))

    def test_noise_band_needs_no_evidence_for_any_trend(self):
        for baseline in (self.expanding, self.contracting, self.mixed):
            m = baseline.base_operating_margin * 100
            r = self._screen(baseline, [m - 0.9, m + 0.9, m - 0.5, m, m])
            self.assertEqual(r.adjustments, [], baseline.margin_trend)

    def test_loss_maker_margin_stays_within_engine_bounds(self):
        deep = self.expanding.model_copy(update={"base_operating_margin": -1.99, "margin_trend": "contracting"})
        r = self._screen(deep, [-250.0] * 5, citation="c1", evidence={"c1"})
        self.assertTrue(all(m >= -2.0 for m in r.drivers.operating_margin))


if __name__ == "__main__":
    unittest.main()
