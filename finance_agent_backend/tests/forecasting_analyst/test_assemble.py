"""Tests for the forecast assembly: code-written reinvestment rationale, warning order and citation screening."""

import unittest

from app.agents.specialized.forecasting_analyst.assemble import assemble_forecast, resolve_forecast_citations
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import build_forecast_baseline
from app.agents.tools.forecast_drivers import DEFAULT_TERMINAL_GROWTH, historical_fade_drivers
from app.agents.tools.forecast_model import build_forecast
from app.agents.specialized.forecasting_analyst.modules.management_guidance import MdnaAssessment
from tests.forecast_fixtures import GUIDANCE_CHUNK, make_aapl_audit, make_chunk


class TestAssembleForecast(unittest.TestCase):
    def setUp(self):
        self.baseline = build_forecast_baseline(make_aapl_audit())
        self.build = build_forecast(
            self.baseline, historical_fade_drivers(self.baseline, DEFAULT_TERMINAL_GROWTH), DEFAULT_TERMINAL_GROWTH
        )

    def _assemble(self, **kwargs):
        return assemble_forecast(
            ticker="aapl", fiscal_year=2025, baseline=self.baseline, build=self.build,
            guidance_source="mdna_informed", mdna=MdnaAssessment(), terminal_growth=DEFAULT_TERMINAL_GROWTH, **kwargs,
        )

    def test_the_reinvestment_rationale_is_always_code_written(self):
        out = self._assemble(rationales={"growth_rationale": "LLM growth.", "reinvestment_rationale": "LLM capex."})
        self.assertEqual(out.growth_rationale, "LLM growth.")
        self.assertTrue(out.reinvestment_rationale.startswith("CapEx starts at"))

    def test_extra_warnings_follow_the_build_warnings(self):
        out = self._assemble(extra_warnings=["LLM step failed."])
        self.assertEqual(out.diagnostics.warnings, self.build.diagnostics.warnings + ["LLM step failed."])
        self.assertEqual(out.ticker, "AAPL")


class TestResolveForecastCitations(unittest.TestCase):
    def test_keeps_only_chunks_the_agent_saw_in_order_without_duplicates(self):
        chunks = [make_chunk("c1", GUIDANCE_CHUNK)]
        cited = resolve_forecast_citations(
            [{"chunk_id": "t9"}, {"chunk_id": "bogus"}, {"chunk_id": "c1"}],
            chunks, ["c1", "t9"], [{"chunk_id": "t9", "item": "Item 8"}],
        )
        self.assertEqual([c["chunk_id"] for c in cited], ["t9", "c1"])
        self.assertEqual(cited[1], {"chunk_id": "c1", "item": "Item 7", "breadcrumb": "Item 7 > MD&A"})


if __name__ == "__main__":
    unittest.main()
