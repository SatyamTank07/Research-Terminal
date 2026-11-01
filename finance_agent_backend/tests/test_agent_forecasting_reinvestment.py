"""Forecasting Analyst wiring for the margin-evidence screen, the CapEx steady state and tool robustness (no LLM)."""

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
from app.agents.specialized.forecasting_analyst.agent_forecasting_analyst import PROPOSE_TOOL_NAME
from app.agents.tools.forecast_baseline import build_forecast_baseline
from tests.forecast_fixtures import (
    AMZN_GUIDANCE_CHUNK,
    GUIDANCE_CHUNK,
    MARGIN_OUTLOOK_CHUNK,
    make_aapl_audit,
    make_chunk,
)

RETRIEVE = "app.agents.specialized.forecasting_analyst.context.retrieve_10k_narrative"
CHUNKS = [make_chunk("c1", GUIDANCE_CHUNK), make_chunk("m1", MARGIN_OUTLOOK_CHUNK)]
REVERSAL = {"revenue_growth_pct": [6.0, 5.5, 5.0, 4.0, 3.0], "operating_margin_pct": [31.0, 29.0, 27.0, 26.0, 25.0]}
FINAL = json.dumps({"growth_rationale": "g", "margin_rationale": "m", "citations": []})


def _run(proposals: List[Dict[str, Any]], chunks=CHUNKS):
    forecaster = ForecastingAnalystAgent()

    def invoke(inputs, config=None):
        tool = next(t for t in forecaster.get_tools() if t.name == PROPOSE_TOOL_NAME)
        messages: List[Any] = [HumanMessage(content="task")]
        for i, proposal in enumerate(proposals):
            payload = tool.invoke(proposal)
            messages.append(ToolMessage(content=json.dumps(payload), name=PROPOSE_TOOL_NAME, tool_call_id=f"c{i}"))
        messages.append(AIMessage(content=FINAL))
        return {"messages": messages}

    llm = MagicMock()
    llm.invoke.side_effect = invoke
    with patch(RETRIEVE, return_value=chunks), \
            patch.object(ForecastingAnalystAgent, "_get_or_create_agent", return_value=llm):
        return forecaster.forecast(ticker="AAPL", fiscal_year=2025, financial_audit=make_aapl_audit())


class TestForecasterMarginEvidence(unittest.TestCase):
    def test_uncited_margin_reversal_is_clamped(self):
        out = _run([REVERSAL])
        self.assertEqual(out.guidance_source, "mdna_informed")
        floor = round(out.assumptions.base_operating_margin * 100 - 1.0, 2)
        self.assertTrue(all(y.projected_ebit_margin_pct >= floor for y in out.forecast_schedule))
        self.assertTrue(any(a.driver == "operating_margin" and "expanding" in a.reason for a in out.driver_adjustments))

    def test_cited_margin_outlook_allows_reversal(self):
        out = _run([{**REVERSAL, "margin_citation_chunk_id": "m1"}])
        self.assertEqual([y.projected_ebit_margin_pct for y in out.forecast_schedule], [31.0, 29.0, 27.0, 26.0, 25.0])
        self.assertFalse(any(a.driver == "operating_margin" for a in out.driver_adjustments))

    def test_capex_guidance_chunk_is_not_margin_evidence(self):
        out = _run([{**REVERSAL, "margin_citation_chunk_id": "c1"}])
        self.assertTrue(any("citation 'c1'" in a.reason for a in out.driver_adjustments))

    def test_reinvestment_rationale_is_code_written_and_describes_the_fade(self):
        out = _run([REVERSAL])
        self.assertIn("steady state", out.reinvestment_rationale)
        self.assertIn("return on new capital", out.reinvestment_rationale)
        d = out.applied_drivers
        self.assertEqual(len(d.capex_path), out.forecast_horizon_years)
        self.assertEqual(d.capex_path[-1], d.steady_state_capex_pct_of_revenue)

    def test_amzn_style_guidance_does_not_trigger_the_llm(self):
        forecaster = ForecastingAnalystAgent()
        with patch(RETRIEVE, return_value=[make_chunk("a1", AMZN_GUIDANCE_CHUNK)]), \
                patch.object(ForecastingAnalystAgent, "_get_or_create_agent") as get_agent:
            out = forecaster.forecast(ticker="AMZN", fiscal_year=2025, financial_audit=make_aapl_audit())
        get_agent.assert_not_called()
        self.assertEqual(out.guidance_source, "historical_fade")


class TestProposeToolNeverRaises(unittest.TestCase):
    def setUp(self):
        self.forecaster = ForecastingAnalystAgent()
        baseline = build_forecast_baseline(make_aapl_audit())
        self.forecaster._configure_tools(baseline, {"c1", "m1"}, {"m1"}, 0.025, None)
        self.tool = next(t for t in self.forecaster.get_tools() if t.name == PROPOSE_TOOL_NAME)

    def test_bad_inputs_come_back_rejected(self):
        for kwargs in (
            dict(revenue_growth_pct=[None, 1, 2, 3, 4], operating_margin_pct=[30] * 5),
            dict(revenue_growth_pct=["fast"] * 5, operating_margin_pct=[30] * 5),
            dict(revenue_growth_pct=[5] * 5, operating_margin_pct=[30] * 3),
            dict(revenue_growth_pct=[float("inf")] * 5, operating_margin_pct=[30] * 5),
            dict(revenue_growth_pct=[5] * 5, operating_margin_pct=[30] * 5, margin_citation_chunk_id=123),
        ):
            result = self.tool.func(**kwargs)
            self.assertEqual(result["status"], "rejected", kwargs)
            self.assertTrue(result["errors"])

    def test_accepted_payload_reports_margin_evidence(self):
        result = self.tool.func(
            revenue_growth_pct=[5] * 5, operating_margin_pct=[30] * 5, margin_citation_chunk_id="m1",
        )
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["margin_evidence_accepted"])
        self.assertIn("capex_pct", result["preview"][0])


if __name__ == "__main__":
    unittest.main()
