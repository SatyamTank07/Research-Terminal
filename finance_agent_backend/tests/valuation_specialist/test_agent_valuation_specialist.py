"""Tests for the deterministic-first Valuation Specialist.

Covers the code-built valuation (Blume beta, Kd from interest over average debt, value-driver terminal
value, NCI bridge, roll-forward, bull/base/bear values), the WACC override and missing-market-data paths,
the single commentary LLM call with its template fallback, and the pipeline AGENT spec wiring. Also the
whole-build checks: reverse DCF, RONIC and implied multiples, company-profile edges and the warning order.
"""

import unittest
from datetime import date
from unittest.mock import patch

from app.agents.query_parameters import parse_valuation_overrides
from app.agents.registry import AgentRegistry
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.valuation_specialist import DCFValuationOutput, ValuationSpecialistAgent
from app.agents.specialized.valuation_specialist.agent_valuation_specialist import build_valuation
from app.agents.specialized.valuation_specialist.context import MarketInputs
from app.agents.specialized.valuation_specialist.modules.cost_of_capital import blume_adjust
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.calculate import (
    calculate_dcf,
    dcf_fair_value_per_share,
)
from app.agents.specialized.forecasting_analyst import project_with_overrides
from app.agents.specialized.valuation_specialist.write import summarize_from_template
from app.config import settings
from tests.forecast_fixtures import (
    make_aapl_audit,
    make_audit,
    make_forecast_output,
    make_heavy_capex_audit,
    make_loss_audit,
    make_shrinking_audit,
)
from tests.valuation_fixtures import AAPL_BALANCE_SHEET_DATE, VALUATION_DATE, make_market, make_valuation_output

COMMENTARY = (
    "Valuation bridge: most of the enterprise value comes from the terminal value, which is typical. "
    "Hurdle rate: the cost of equity is driven by beta and the equity risk premium. "
    "Range: the bear and bull cases bracket the base case. "
    "Margin of safety: the base case sits close to the market price."
)


def _indebted_audit():
    rows = [
        dict(fiscal_year=2024, revenue=19000.0, gross_profit=9500.0, operating_income=4600.0,
             net_income=2800.0, operating_cash_flow=3800.0, capital_expenditures=950.0, interest_expense=950.0),
        dict(fiscal_year=2025, revenue=20000.0, gross_profit=10000.0, operating_income=5000.0,
             net_income=3000.0, operating_cash_flow=4000.0, capital_expenditures=1000.0, interest_expense=1050.0),
    ]
    balance = dict(fiscal_year=2025, cash_and_equivalents=2000.0, marketable_securities=1000.0,
                   short_term_debt=2000.0, long_term_debt=18000.0, stockholders_equity=10000.0,
                   noncontrolling_interests=800.0, weighted_diluted_shares=500.0)
    return make_audit("IND", rows, balance)


class TestValuationSpecialistAgent(unittest.TestCase):
    def test_01_registry_and_prompts(self):
        agent = AgentRegistry.get("valuation_specialist")
        self.assertIsInstance(agent, ValuationSpecialistAgent)
        self.assertEqual(agent.get_tools(), [])

        system = render_prompt("valuation_specialist", "prompt_valuation_specialist.j2")
        self.assertIn("Use ONLY numbers that appear in the facts block", system)
        self.assertIn("four short paragraphs", system)

        v = make_valuation_output()
        query = render_prompt("valuation_specialist", "write.j2", v=v)
        self.assertIn(f"${v.implied_fair_value_per_share:.2f}/share", query)
        self.assertIn("probability-weighted", query)
        self.assertIn(v.wacc_audit.risk_free_rate_source, query)
        self.assertIn("*(Base)*", query)

    @patch.object(ValuationSpecialistAgent, "_run_commentary_llm", return_value=COMMENTARY)
    def test_02_aapl_valuation_is_code_built(self, mock_llm):
        audit, forecast, market = make_aapl_audit(), make_forecast_output(), make_market()
        out = ValuationSpecialistAgent().value(
            ticker="AAPL", fiscal_year=2025, financial_audit=audit, forecast=forecast, market=market,
            terminal_growth_rate=0.025, valuation_date=VALUATION_DATE, balance_sheet_date=AAPL_BALANCE_SHEET_DATE,
        )
        self.assertIsInstance(out, DCFValuationOutput)
        self.assertEqual(out.commentary_source, "llm")
        self.assertEqual(out.valuation_summary, COMMENTARY)
        mock_llm.assert_called_once()

        w = out.wacc_audit
        self.assertAlmostEqual(w.beta, round(blume_adjust(1.10), 4))
        self.assertEqual(w.beta_raw, 1.10)
        self.assertEqual(w.risk_free_rate_pct, 4.5)
        self.assertEqual(w.wacc_source, "capm_blended")
        self.assertEqual(w.cost_of_debt_source, "institutional_credit_spread_fallback")  # AAPL fixture: no interest line
        # The debt tax shield and the terminal NOPAT use the marginal rate; explicit years keep the forecast's rate.
        self.assertAlmostEqual(w.effective_tax_rate_pct, settings.DEFAULT_MARGINAL_TAX_RATE * 100.0, places=2)
        self.assertAlmostEqual(out.terminal_tax_rate_pct, settings.DEFAULT_MARGINAL_TAX_RATE * 100.0, places=2)

        # Every headline figure equals an independent engine run with the same inputs.
        roll = (VALUATION_DATE - date(2025, 9, 27)).days / 365.25
        expected = calculate_dcf(
            forecast.projected_fcfs, w.wacc, 0.025, audit.balance_sheet.net_debt,
            audit.balance_sheet.valuation_shares_outstanding,
            terminal_nopat=round(forecast.forecast_schedule[-1].projected_ebit * (1.0 - settings.DEFAULT_MARGINAL_TAX_RATE), 2),
            ronic=out.ronic_used_pct / 100.0, roll_forward_years=roll, risk_free_rate=w.risk_free_rate_pct / 100.0,
        )
        self.assertAlmostEqual(out.implied_fair_value_per_share, expected.fair_value_per_share, delta=0.05)
        self.assertAlmostEqual(out.enterprise_value, expected.enterprise_value, delta=0.0005 * expected.enterprise_value)
        self.assertEqual(out.projected_fcfs, forecast.projected_fcfs)
        self.assertEqual(out.terminal_method, "value_driver")
        self.assertGreaterEqual(out.ronic_used_pct, w.wacc_pct - 0.01)
        self.assertLessEqual(out.ronic_used_pct, 30.0)
        self.assertAlmostEqual(out.roll_forward_years, roll, places=3)
        self.assertEqual(out.balance_sheet_date, AAPL_BALANCE_SHEET_DATE)

        # Scenarios: base reuses the headline, bear < base < bull, weighted = sum(p x value)
        names = [s.name for s in out.scenario_valuations]
        self.assertEqual(names, ["bull", "base", "bear"])
        by = {s.name: s for s in out.scenario_valuations}
        self.assertEqual(by["base"].fair_value_per_share, out.implied_fair_value_per_share)
        self.assertLess(out.bear_fair_value, out.implied_fair_value_per_share)
        self.assertGreater(out.bull_fair_value, out.implied_fair_value_per_share)
        weighted = sum(s.probability * s.fair_value_per_share for s in out.scenario_valuations)
        self.assertAlmostEqual(out.probability_weighted_fair_value, weighted, places=2)
        self.assertIn("Probability-weighted", out.scenario_table_markdown)

        # Price-based fields are on the base case
        self.assertEqual(out.current_share_price, 235.0)
        self.assertAlmostEqual(out.upside_downside_pct, round((out.implied_fair_value_per_share - 235.0) / 2.35, 2))
        self.assertIn(out.valuation_stance, ("Undervalued", "Fairly Valued", "Overvalued"))
        self.assertAlmostEqual(out.implied_ev_ebitda, round(out.enterprise_value / (133050.0 + 11445.0), 2))
        self.assertIn("Equity value at valuation date", out.equity_bridge_markdown)

    @patch.object(ValuationSpecialistAgent, "_run_commentary_llm", return_value=COMMENTARY)
    def test_03_indebted_firm_deducts_nci_and_derives_kd(self, _llm):
        audit = _indebted_audit()
        out = ValuationSpecialistAgent().value(
            ticker="IND", fiscal_year=2025, financial_audit=audit, forecast=None,
            market=make_market(share_price=45.0, beta=1.2), terminal_growth_rate=0.025,
            valuation_date=VALUATION_DATE, balance_sheet_date=None,
        )
        self.assertEqual(out.net_debt, 17000.0)
        self.assertEqual(out.noncontrolling_interests, 800.0)
        self.assertAlmostEqual(out.equity_value, out.enterprise_value - 17000.0 - 800.0, places=1)
        self.assertEqual(out.roll_forward_years, 0.0)
        self.assertIn("Balance-sheet date unknown", " ".join(out.data_quality.warnings))

        # Kd = 1,050 / 20,000 (prior debt unknown => ending debt) = 5.25%
        self.assertEqual(out.wacc_audit.cost_of_debt_source, "derived_from_10k_interest_expense")
        self.assertAlmostEqual(out.wacc_audit.cost_of_debt_pre_tax_pct, 5.25, places=2)

        # No forecast => flagged naive fallback, no scenarios; no D&A => no EV/EBITDA
        self.assertEqual(out.data_quality.forecast_source, "naive_5pct_fallback")
        self.assertEqual(out.scenario_valuations, [])
        self.assertIsNone(out.probability_weighted_fair_value)
        self.assertEqual(out.terminal_method, "last_fcf_growth")
        self.assertIsNone(out.implied_ev_ebitda)

    def test_04_wacc_override_keeps_capm_components(self):
        out = make_valuation_output(wacc_override=0.09)
        w = out.wacc_audit
        self.assertEqual(w.wacc, 0.09)
        self.assertEqual(w.wacc_source, "user_override")
        self.assertNotEqual(w.capm_wacc_pct, 9.0)
        self.assertGreater(w.debt_weight_pct, 0.0)
        self.assertGreater(w.cost_of_debt_pre_tax_pct, 0.0)
        self.assertIn("User-specified override", w.formula_breakdown_markdown)

    def test_05_missing_market_data_is_flagged(self):
        out = make_valuation_output(market=MarketInputs())
        self.assertEqual(out.wacc_audit.beta, 1.0)
        self.assertEqual(out.wacc_audit.beta_source, "default_market_beta")
        self.assertEqual(out.wacc_audit.capital_structure_source, "book_equity_fallback")
        self.assertIsNone(out.current_share_price)
        self.assertIsNone(out.valuation_stance)
        self.assertIsNone(out.upside_downside_pct)
        joined = " ".join(out.data_quality.warnings)
        self.assertIn("configured default", joined)
        self.assertIn("market beta of 1.0", joined)
        self.assertIn("book equity", joined)

    @patch.object(ValuationSpecialistAgent, "_run_commentary_llm", side_effect=RuntimeError("rate limited"))
    def test_06_commentary_failure_falls_back_to_template(self, _llm):
        out = ValuationSpecialistAgent().value(
            ticker="AAPL", fiscal_year=2025, financial_audit=make_aapl_audit(), forecast=make_forecast_output(),
            market=make_market(), terminal_growth_rate=0.025, valuation_date=VALUATION_DATE,
            balance_sheet_date=AAPL_BALANCE_SHEET_DATE,
        )
        self.assertEqual(out.commentary_source, "template")
        self.assertIn(f"${out.implied_fair_value_per_share:.2f} per share", out.valuation_summary)
        self.assertIn("probability-weighted", out.valuation_summary)

    def test_07_status_message(self):
        from app.agents.specialized.valuation_specialist import AGENT

        out = make_valuation_output()
        msg, _ = AGENT.milestone(out)
        self.assertIn(f"${out.implied_fair_value_per_share:.2f}/share", msg)
        self.assertIn(f"bear ${out.bear_fair_value:.2f}", msg)
        self.assertIn("CAPM WACC", msg)
        self.assertEqual(AGENT.milestone(None), ("Valuation unavailable", None))


class TestValuationSpecWiring(unittest.TestCase):
    CONTEXT = "app.agents.specialized.valuation_specialist.context"
    AGENT_MODULE = "app.agents.specialized.valuation_specialist.agent_valuation_specialist"

    def _run(self, query: str, market_data: dict, **ctx_changes):
        from app.agents.pipeline.contract import Upstream
        from app.agents.specialized.valuation_specialist import AGENT
        from tests.pipeline.fixtures import make_run_context

        ctx = make_run_context(valuation_overrides=parse_valuation_overrides(query).values(), **ctx_changes)
        upstream = Upstream({"financial_audit": make_aapl_audit(), "forecast": make_forecast_output()}, AGENT.declared)
        with patch(f"{self.CONTEXT}.fetch_market_context", return_value=market_data), \
                patch(f"{self.AGENT_MODULE}.balance_sheet_date", return_value=AAPL_BALANCE_SHEET_DATE), \
                patch.object(ValuationSpecialistAgent, "value", return_value="sentinel") as mock_value:
            self.assertEqual(AGENT.run(ctx, upstream), "sentinel")
        return ctx, mock_value.call_args.kwargs

    def test_user_overrides_take_precedence(self):
        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": 1.1, "beta_source": "yfinance_raw",
                "risk_free_rate": 0.044, "risk_free_rate_source": "us_10y_treasury_live",
                "market_data_source": "yfinance_live"}
        ctx, kwargs = self._run("value AAPL with beta 1.3, share price 200 and WACC of 9%", live)
        market: MarketInputs = kwargs["market"]
        self.assertEqual((market.beta, market.beta_source), (1.3, "user_provided"))
        self.assertEqual((market.share_price, market.share_price_source), (200.0, "user_provided"))
        self.assertIsNone(market.market_cap)  # live cap is inconsistent with a user price
        self.assertEqual(market.risk_free_rate, 0.044)
        self.assertAlmostEqual(kwargs["wacc_override"], 0.09)
        self.assertEqual(kwargs["balance_sheet_date"], AAPL_BALANCE_SHEET_DATE)
        self.assertEqual(kwargs["valuation_date"], ctx.valuation_date)
        self.assertIsNotNone(kwargs["forecast"])

    def test_live_market_data_passes_through(self):
        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": None, "beta_source": "unavailable",
                "risk_free_rate": 0.042, "risk_free_rate_source": "house_default_config",
                "market_data_source": "yfinance_live"}
        _, kwargs = self._run("DCF for AAPL", live)
        market: MarketInputs = kwargs["market"]
        self.assertEqual(market.share_price_source, "yfinance_live")
        self.assertEqual(market.market_cap, 3_450_000.0)
        self.assertIsNone(market.beta)
        self.assertEqual(market.risk_free_rate_source, "house_default_config")

    def test_forecast_and_valuation_share_one_terminal_growth(self):
        from app.agents.pipeline.contract import Upstream
        from app.agents.specialized.forecasting_analyst import AGENT as FORECASTER
        from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
        from tests.pipeline.fixtures import make_run_context

        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": 1.1, "beta_source": "yfinance_raw",
                "risk_free_rate": 0.044, "risk_free_rate_source": "us_10y_treasury_live",
                "market_data_source": "yfinance_live"}
        ctx, valuation_kwargs = self._run("DCF for AAPL", live, terminal_growth_rate=0.031)
        upstream = Upstream({"financial_audit": make_aapl_audit()}, FORECASTER.declared)
        with patch.object(ForecastingAnalystAgent, "forecast", return_value="forecast") as mock_forecast:
            FORECASTER.run(ctx, upstream)
        self.assertEqual(mock_forecast.call_args.kwargs["terminal_growth"], valuation_kwargs["terminal_growth_rate"])
        self.assertAlmostEqual(valuation_kwargs["terminal_growth_rate"], 0.031)


class TestBuildValuationEdges(unittest.TestCase):
    def test_low_ronic_is_floored_at_wacc_and_warned(self):
        forecast = make_forecast_output()
        diag = forecast.diagnostics.model_copy(update={"steady_state_ronic_pct": 2.0})
        build = build_valuation(
            "AAPL", 2025, make_aapl_audit(), forecast.model_copy(update={"diagnostics": diag}), make_market(),
            terminal_growth_rate=0.025, valuation_date=VALUATION_DATE, balance_sheet_date=AAPL_BALANCE_SHEET_DATE,
        )
        self.assertAlmostEqual(build.ronic_used_pct, build.wacc_audit.wacc * 100.0, places=2)
        self.assertIn("clamped", " ".join(build.data_quality.warnings))



class TestCommentaryCallSettings(unittest.TestCase):
    def test_commentary_call_has_system_prompt_cap_and_five_retries(self):
        from app.agents.specialized.valuation_specialist.agent_valuation_specialist import COMMENTARY_MAX_TOKENS
        agent = ValuationSpecialistAgent()
        with patch("app.agents.specialized.valuation_specialist.agent_valuation_specialist.complete_once", return_value="x") as call:
            agent._run_commentary_llm("p", None)
        call.assert_called_once_with(
            agent.model_name, "p", system_prompt=agent.get_system_prompt(),
            max_tokens=COMMENTARY_MAX_TOKENS, max_retries=5, callbacks=None,
        )


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
        query = render_prompt("valuation_specialist", "write.j2", v=out)
        self.assertIn("- Reverse DCF:", query)
        self.assertIn("WACC rows x terminal RONIC columns", query)
        self.assertIn("EV/NOPAT", query)
        self.assertIn("WACC x RONIC", summarize_from_template(out))


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


class TestValuationWarningOrder(unittest.TestCase):
    def test_no_forecast_warning_order_is_placeholder_then_capital_then_value(self):
        out = build_valuation(
            "AAPL", 2025, make_aapl_audit(), None, MarketInputs(),
            terminal_growth_rate=0.025, valuation_date=VALUATION_DATE, balance_sheet_date=None,
        )
        w = out.data_quality.warnings
        self.assertTrue(w[0].startswith("No forecast available"))
        self.assertTrue(w[1].startswith("No beta available"))
        self.assertTrue(w[2].startswith("Risk-free rate"))
        roll = next(i for i, x in enumerate(w) if x.startswith("Balance-sheet date unknown"))
        self.assertGreater(roll, 2)
        self.assertEqual(out.data_quality.forecast_source, "naive_5pct_fallback")


if __name__ == "__main__":
    unittest.main()
