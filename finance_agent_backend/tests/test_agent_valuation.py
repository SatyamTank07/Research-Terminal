"""Tests for the deterministic-first Valuation Specialist.

Covers the code-built valuation (Blume beta, Kd from interest over average debt, value-driver terminal
value, NCI bridge, roll-forward, bull/base/bear values), the WACC override and missing-market-data paths,
the single commentary LLM call with its template fallback, and the orchestrator node wiring.
"""

import asyncio
import unittest
from datetime import date
from unittest.mock import patch

from app.agents.orchestrator import MultiAgentOrchestrator, valuation_specialist_node
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.valuation_specialist import DCFValuationOutput, ValuationSpecialistAgent
from app.agents.tools.dcf_tools import calculate_dcf_with_sensitivity
from app.agents.tools.valuation_model import MarketInputs, build_valuation
from app.agents.tools.wacc_tools import blume_adjust
from tests.forecast_fixtures import make_aapl_audit, make_audit, make_forecast_output
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

        system = render_prompt("valuation_specialist")
        self.assertIn("Use ONLY numbers that appear in the facts block", system)
        self.assertIn("four short paragraphs", system)

        v = make_valuation_output()
        query = render_prompt("prompt_valuation_query.j2", v=v)
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
        self.assertAlmostEqual(w.effective_tax_rate_pct, forecast.tax_rate_pct, places=2)

        # Every headline figure equals an independent engine run with the same inputs.
        roll = (VALUATION_DATE - date(2025, 9, 27)).days / 365.25
        expected = calculate_dcf_with_sensitivity(
            forecast.projected_fcfs, w.wacc, 0.025, audit.balance_sheet.net_debt,
            audit.balance_sheet.valuation_shares_outstanding,
            terminal_nopat=forecast.forecast_schedule[-1].projected_nopat,
            ronic=out.ronic_used_pct / 100.0, roll_forward_years=roll, cost_of_equity=w.cost_of_equity,
        )
        self.assertAlmostEqual(out.implied_fair_value_per_share, expected["fair_value_per_share"], delta=0.05)
        self.assertAlmostEqual(out.enterprise_value, expected["enterprise_value"], delta=0.0005 * expected["enterprise_value"])
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
        out = make_valuation_output()
        msg = MultiAgentOrchestrator._valuation_status_message(out)
        self.assertIn(f"${out.implied_fair_value_per_share:.2f}/share", msg)
        self.assertIn(f"bear ${out.bear_fair_value:.2f}", msg)
        self.assertIn("CAPM WACC", msg)
        self.assertEqual(MultiAgentOrchestrator._valuation_status_message(None), "Valuation unavailable")


class TestValuationNodeWiring(unittest.TestCase):
    def _run_node(self, query: str, market_data: dict):
        state = {
            "ticker": "AAPL", "fiscal_year": 2025, "financial_audit": make_aapl_audit(),
            "forecast": make_forecast_output(), "user_query": query, "callbacks": None, "session_state": {},
        }
        with patch("app.agents.orchestrator.fetch_market_context", return_value=market_data), \
                patch("app.agents.orchestrator._balance_sheet_date", return_value=AAPL_BALANCE_SHEET_DATE), \
                patch.object(ValuationSpecialistAgent, "value", return_value="sentinel") as mock_value:
            result = asyncio.run(valuation_specialist_node(state))
        self.assertEqual(result, {"dcf_valuation": "sentinel"})
        return mock_value.call_args.kwargs

    def test_user_overrides_take_precedence(self):
        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": 1.1, "beta_source": "yfinance_raw",
                "risk_free_rate": 0.044, "risk_free_rate_source": "us_10y_treasury_live",
                "market_data_source": "yfinance_live"}
        kwargs = self._run_node("value AAPL with beta 1.3, share price 200 and WACC of 9%", live)
        market: MarketInputs = kwargs["market"]
        self.assertEqual((market.beta, market.beta_source), (1.3, "user_provided"))
        self.assertEqual((market.share_price, market.share_price_source), (200.0, "user_provided"))
        self.assertIsNone(market.market_cap)  # live cap is inconsistent with a user price
        self.assertEqual(market.risk_free_rate, 0.044)
        self.assertAlmostEqual(kwargs["wacc_override"], 0.09)
        self.assertEqual(kwargs["balance_sheet_date"], AAPL_BALANCE_SHEET_DATE)
        self.assertIsInstance(kwargs["valuation_date"], date)
        self.assertIsNotNone(kwargs["forecast"])

    def test_live_market_data_passes_through(self):
        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": None, "beta_source": "unavailable",
                "risk_free_rate": 0.042, "risk_free_rate_source": "house_default_config",
                "market_data_source": "yfinance_live"}
        market: MarketInputs = self._run_node("DCF for AAPL", live)["market"]
        self.assertEqual(market.share_price_source, "yfinance_live")
        self.assertEqual(market.market_cap, 3_450_000.0)
        self.assertIsNone(market.beta)
        self.assertEqual(market.risk_free_rate_source, "house_default_config")


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


if __name__ == "__main__":
    unittest.main()
