"""Tests for the Forecasting Analyst agent (mocked LLM, no network)."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.orchestrator import MultiAgentOrchestrator, _resolve_terminal_growth, forecasting_analyst_node
from app.agents.registry import AgentRegistry
from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
from app.agents.specialized.forecasting_analyst.agent_forecasting_analyst import (
    LLM_STEP_FAILED_WARNING,
    NO_ACCEPTED_DRIVERS_WARNING,
    PROPOSE_TOOL_NAME,
)
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.valuation_specialist import ValuationSpecialistAgent
from app.agents.state import DCFValuationOutput, WACCAudit
from app.agents.tools.dcf_tools import calculate_dcf_with_sensitivity
from app.agents.tools.forecast_baseline import build_forecast_baseline
from app.agents.tools.forecast_drivers import DEFAULT_TERMINAL_GROWTH
from app.agents.tools.mdna_guidance import assess_mdna_guidance
from app.agents.tools.wacc_tools import calculate_wacc
from tests.forecast_fixtures import (
    BOILERPLATE_CHUNK,
    GUIDANCE_CHUNK,
    HISTORY_CHUNK,
    make_aapl_audit,
    make_chunk,
    make_forecast_output,
)

RETRIEVE = "app.agents.specialized.forecasting_analyst.agent_forecasting_analyst.retrieve_10k_narrative"
INFORMATIVE = [make_chunk("c1", GUIDANCE_CHUNK), make_chunk("c2", HISTORY_CHUNK)]
BOILERPLATE = [make_chunk("b1", BOILERPLATE_CHUNK)]
RATIONALES: Dict[str, Any] = {
    "growth_rationale": "Management expects Services to outgrow Products [c1]; growth fades from 7.0% to 4.0%.",
    "margin_rationale": "Margin rises from 31.97% to 33.0% on Services mix.",
    "reinvestment_rationale": "CapEx lifted to 3.5% of revenue on the stated $14 billion data-center plan [c1].",
    "citations": [{"chunk_id": "c1", "item": "Item 7"}, {"chunk_id": "bogus"}],
}
GOOD_PROPOSAL: Dict[str, Any] = {
    "revenue_growth_pct": [7.0, 6.5, 6.0, 5.0, 4.0],
    "operating_margin_pct": [32.0, 32.5, 33.0, 33.0, 33.0],
    "capex_pct_of_revenue": 3.5,
    "capex_citation_chunk_id": "c1",
    "evidence_chunk_ids": ["c1", "bogus"],
}


def _mock_llm(forecaster: ForecastingAnalystAgent, proposals: List[Dict[str, Any]], final_content: str) -> MagicMock:
    """Mocked LangChain agent: calls the per-run driver tool once per proposal, then answers."""

    def invoke(inputs, config=None):
        tool = next(t for t in forecaster.get_tools() if t.name == PROPOSE_TOOL_NAME)
        messages: List[Any] = [HumanMessage(content="task")]
        for i, proposal in enumerate(proposals):
            payload = tool.invoke(proposal)
            messages.append(ToolMessage(content=json.dumps(payload), name=PROPOSE_TOOL_NAME, tool_call_id=f"call-{i}"))
        messages.append(AIMessage(content=final_content))
        return {"messages": messages}

    agent = MagicMock()
    agent.invoke.side_effect = invoke
    return agent


def _create_mock_valuation_agent(projected_fcfs: List[float]) -> MagicMock:
    wacc_res = calculate_wacc(
        beta=1.24, total_debt=98657.0, market_cap=230.0 * 15004.7,
        risk_free_rate=0.042, equity_risk_premium=0.05, tax_rate=0.1561,
    )
    dcf_res = calculate_dcf_with_sensitivity(
        projected_fcfs=projected_fcfs, wacc=wacc_res["wacc"], terminal_growth_rate=0.025,
        net_debt=-33763.0, diluted_shares=15004.7, mid_year_convention=True,
    )
    mock_output = DCFValuationOutput(
        ticker="AAPL", fiscal_year=2025, wacc_audit=WACCAudit.model_validate(wacc_res),
        terminal_growth_rate=0.025, discounting_convention="mid_year", projected_fcfs=projected_fcfs,
        pv_explicit_fcfs=dcf_res["pv_explicit_fcfs"], pv_terminal_value=dcf_res["pv_terminal_value"],
        terminal_value_pct_of_ev=dcf_res["terminal_value_pct_of_ev"], enterprise_value=dcf_res["enterprise_value"],
        net_debt=dcf_res["net_debt"], equity_value=dcf_res["equity_value"], diluted_shares=dcf_res["diluted_shares"],
        implied_fair_value_per_share=dcf_res["fair_value_per_share"], current_share_price=230.0,
        upside_downside_pct=round(((dcf_res["fair_value_per_share"] - 230.0) / 230.0) * 100.0, 2),
        valuation_stance="Fairly Valued", sensitivity_matrix_markdown=dcf_res["sensitivity_matrix_markdown"],
        valuation_summary="DCF valuation completed for Apple Inc. FY2025 based on explicit projections.",
    )
    agent = MagicMock()
    agent.invoke.return_value = {
        "messages": [MagicMock(content="Valuation completed", tool_calls=[])],
        "structured_response": mock_output,
    }
    return agent


class TestForecastingAnalystAgent(unittest.TestCase):
    def setUp(self):
        self.audit = make_aapl_audit()

    def _run(self, proposals, final_content=json.dumps(RATIONALES), **kwargs):
        forecaster = ForecastingAnalystAgent()
        llm = _mock_llm(forecaster, proposals, final_content)
        with patch(RETRIEVE, return_value=INFORMATIVE), \
                patch.object(ForecastingAnalystAgent, "_get_or_create_agent", return_value=llm):
            return forecaster.forecast(ticker="AAPL", fiscal_year=2025, financial_audit=self.audit, **kwargs)

    def test_registry_and_system_prompt(self):
        self.assertIsInstance(AgentRegistry.get("forecasting_analyst"), ForecastingAnalystAgent)
        prompt = render_prompt("forecasting_analyst")
        self.assertIn(PROPOSE_TOOL_NAME, prompt)
        self.assertIn("WHOLE percent", prompt)

    def test_query_prompt_carries_baseline_guidance_and_risks(self):
        risk = SimpleNamespace(
            overall_risk_profile="High",
            identified_risks=[SimpleNamespace(
                risk_id="R1", severity="Severe", risk_title="China demand slowdown", financial_transmission=["Revenue"],
            )],
        )
        query = ForecastingAnalystAgent()._render_query(
            "AAPL", 2025, build_forecast_baseline(self.audit), assess_mdna_guidance(INFORMATIVE),
            INFORMATIVE, None, risk, 0.025,
        )
        self.assertIn("416,161", query)
        self.assertIn("We expect capital expenditures of approximately $14 billion", query)
        self.assertIn("R1 [Severe] China demand slowdown (hits: Revenue)", query)
        self.assertIn("[c2]", query)
        self.assertIn("2.50%", query)
        self.assertNotIn("%%", query)

    def test_boilerplate_mdna_skips_the_llm(self):
        forecaster = ForecastingAnalystAgent()
        with patch(RETRIEVE, return_value=BOILERPLATE), \
                patch.object(ForecastingAnalystAgent, "_get_or_create_agent") as get_agent:
            out = forecaster.forecast(ticker="AAPL", fiscal_year=2025, financial_audit=self.audit)
        get_agent.assert_not_called()
        self.assertEqual(out.guidance_source, "historical_fade")
        self.assertEqual(out.forecast_horizon_years, 5)
        self.assertEqual(out.forecast_schedule[-1].projected_revenue_growth_pct, 2.5)
        self.assertEqual([s.name for s in out.scenarios], ["bull", "base", "bear"])
        self.assertEqual(out.projected_fcfs, out.scenarios[1].projected_fcfs)
        self.assertIn("forward-looking", out.growth_rationale)
        self.assertEqual(out.assumptions.da_source, "reported")

    def test_llm_drivers_are_screened_and_rebuilt_in_code(self):
        out = self._run([GOOD_PROPOSAL])
        self.assertEqual(out.guidance_source, "mdna_informed")
        self.assertEqual([y.projected_revenue_growth_pct for y in out.forecast_schedule], [7.0, 6.5, 6.0, 5.0, 4.0])
        self.assertEqual([y.projected_ebit_margin_pct for y in out.forecast_schedule], [32.0, 32.5, 33.0, 33.0, 33.0])
        y1 = out.forecast_schedule[0]
        self.assertAlmostEqual(y1.projected_capex / y1.projected_revenue, 0.035, places=4)
        self.assertEqual(out.driver_adjustments, [])
        self.assertEqual(out.growth_rationale, RATIONALES["growth_rationale"])
        self.assertEqual([c["chunk_id"] for c in out.citations], ["c1"])
        self.assertTrue(out.mdna_assessment.is_informative)

    def test_out_of_band_drivers_are_clamped_and_logged(self):
        out = self._run([{**GOOD_PROPOSAL, "revenue_growth_pct": [40.0, 8.0, 7.0, 6.0, 5.0]}])
        self.assertEqual(out.forecast_schedule[0].projected_revenue_growth_pct, 21.43)
        adj = out.driver_adjustments[0]
        self.assertEqual((adj.driver, adj.year, adj.action), ("revenue_growth", 1, "clamped"))

    def test_without_accepted_drivers_falls_back_to_historical_fade(self):
        out = self._run([{**GOOD_PROPOSAL, "revenue_growth_pct": [7.0, 6.0, 5.0, 4.0]}])
        self.assertEqual(out.guidance_source, "historical_fade")
        self.assertIn(NO_ACCEPTED_DRIVERS_WARNING, out.diagnostics.warnings)
        self.assertNotEqual(out.growth_rationale, RATIONALES["growth_rationale"])

    def test_last_accepted_call_wins(self):
        second = {**GOOD_PROPOSAL, "revenue_growth_pct": [6.0, 5.5, 5.0, 4.5, 4.0]}
        out = self._run([GOOD_PROPOSAL, second])
        self.assertEqual([y.projected_revenue_growth_pct for y in out.forecast_schedule], [6.0, 5.5, 5.0, 4.5, 4.0])

    def test_unparseable_final_answer_gets_code_written_rationales(self):
        out = self._run([GOOD_PROPOSAL], final_content="I am done.")
        self.assertEqual(out.guidance_source, "mdna_informed")
        self.assertTrue(out.growth_rationale.startswith("Revenue growth runs from 7.00%"))
        self.assertEqual([c["chunk_id"] for c in out.citations], ["c1"])

    def _run_failing(self, proposals):
        """LLM step that submits `proposals` through the real tool, then dies (e.g. recursion limit)."""
        forecaster = ForecastingAnalystAgent()

        def invoke(inputs, config=None):
            tool = next(t for t in forecaster.get_tools() if t.name == PROPOSE_TOOL_NAME)
            for proposal in proposals:
                tool.invoke(proposal)
            raise RuntimeError("Recursion limit of 20 reached")

        llm = MagicMock()
        llm.invoke.side_effect = invoke
        with patch(RETRIEVE, return_value=INFORMATIVE), \
                patch.object(ForecastingAnalystAgent, "_get_or_create_agent", return_value=llm):
            return forecaster.forecast(ticker="AAPL", fiscal_year=2025, financial_audit=self.audit)

    def test_llm_failure_after_accepted_call_keeps_screened_drivers(self):
        out = self._run_failing([GOOD_PROPOSAL])
        self.assertEqual(out.guidance_source, "mdna_informed")
        self.assertEqual([y.projected_revenue_growth_pct for y in out.forecast_schedule], [7.0, 6.5, 6.0, 5.0, 4.0])
        self.assertEqual([c["chunk_id"] for c in out.citations], ["c1"])
        self.assertIn(LLM_STEP_FAILED_WARNING, out.diagnostics.warnings)

    def test_llm_failure_without_accepted_call_falls_back_to_fade(self):
        out = self._run_failing([])
        self.assertEqual(out.guidance_source, "historical_fade")
        self.assertIn(LLM_STEP_FAILED_WARNING, out.diagnostics.warnings)
        self.assertIn(NO_ACCEPTED_DRIVERS_WARNING, out.diagnostics.warnings)

    def test_high_risk_profile_deepens_bear_case(self):
        forecaster = ForecastingAnalystAgent()
        with patch(RETRIEVE, return_value=BOILERPLATE):
            calm = forecaster.forecast(ticker="AAPL", fiscal_year=2025, financial_audit=self.audit)
            risky = forecaster.forecast(
                ticker="AAPL", fiscal_year=2025, financial_audit=self.audit,
                risk_audit=SimpleNamespace(overall_risk_profile="High", identified_risks=[]),
            )
        self.assertLess(risky.scenarios[2].cumulative_fcf, calm.scenarios[2].cumulative_fcf)
        self.assertEqual(risky.projected_fcfs, calm.projected_fcfs)

    @patch.object(ValuationSpecialistAgent, "_get_or_create_agent")
    def test_forecast_feeds_valuation(self, mock_val_agent):
        with patch(RETRIEVE, return_value=BOILERPLATE):
            forecast = ForecastingAnalystAgent().forecast(ticker="AAPL", fiscal_year=2025, financial_audit=self.audit)
        mock_val_agent.return_value = _create_mock_valuation_agent(forecast.projected_fcfs)
        val = ValuationSpecialistAgent().value(
            ticker="AAPL", fiscal_year=2025, financial_audit=self.audit,
            projected_fcfs=forecast.projected_fcfs, beta=1.24, share_price=230.0,
        )
        self.assertIsInstance(val, DCFValuationOutput)
        self.assertEqual(val.projected_fcfs, forecast.projected_fcfs)
        self.assertGreater(val.implied_fair_value_per_share, 50.0)


class TestForecastNodeWiring(unittest.TestCase):
    def test_resolve_terminal_growth(self):
        self.assertAlmostEqual(_resolve_terminal_growth("DCF for AMZN with terminal growth 3%"), 0.03)
        self.assertAlmostEqual(_resolve_terminal_growth("terminal growth 0.5%"), 0.005)
        self.assertAlmostEqual(_resolve_terminal_growth("perpetual growth 0.02"), 0.02)
        self.assertEqual(_resolve_terminal_growth("value AMZN"), DEFAULT_TERMINAL_GROWTH)

    def test_node_passes_risk_and_terminal_growth(self):
        forecast = make_forecast_output()
        risk = SimpleNamespace(overall_risk_profile="High", identified_risks=[])
        state = {
            "ticker": "AAPL", "fiscal_year": 2025, "financial_audit": make_aapl_audit(), "business_moat": None,
            "risk_audit": risk, "user_query": "full report on AAPL with terminal growth 3%", "callbacks": None,
        }
        with patch.object(ForecastingAnalystAgent, "forecast", return_value=forecast) as mock_forecast:
            result = asyncio.run(forecasting_analyst_node(state))
        self.assertIs(result["forecast"], forecast)
        kwargs = mock_forecast.call_args.kwargs
        self.assertIs(kwargs["risk_audit"], risk)
        self.assertAlmostEqual(kwargs["terminal_growth"], 0.03)

    def test_status_message(self):
        msg = MultiAgentOrchestrator._forecast_status_message(make_forecast_output())
        self.assertIn("5-year UFCF forecast", msg)
        self.assertIn("historical fade", msg)
        self.assertIn("3 scenarios", msg)
        self.assertEqual(MultiAgentOrchestrator._forecast_status_message(None), "Forecast unavailable")


if __name__ == "__main__":
    unittest.main()
