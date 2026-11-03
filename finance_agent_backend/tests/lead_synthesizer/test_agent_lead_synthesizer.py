"""Unit and Integration Tests for Lead Synthesizer Agent (Milestone 6).

Validates:
1. Prompt template rendering.
2. Full multi-agent synthesis:
   - Ingestion of Moat, Audit, Forecast, DCF Valuation, and Risk payloads.
   - LLM generation of the 3-Pillar Investment Thesis and Executive Summary.
   - Deterministic publication of complete Markdown institutional report.
   - Zero math distortion: verbatim preservation of audited tables and DCF sensitivity matrix.
   - Provenance tracking (synthesis_provenance, year_substituted disclosure).
   - Cross-agent citation deduplication.
3. Graceful handling of partial inputs (e.g. Fast-Path DCF mode without qualitative agents).
4. Exhaustive None-guard resilience: verifying formatting does not crash on None/NaN metrics.
"""

import unittest
from unittest.mock import patch

from app.agents.shared.prompts import render_prompt
from app.agents.specialized.lead_synthesizer import ThreePillarThesis
from app.agents.specialized.lead_synthesizer.agent_lead_synthesizer import LeadSynthesizerAgent
from tests.lead_synthesizer.report_fixtures import (
    FAIR,
    FULL_PROSE,
    TV_SHARE,
    failing_llm,
    mock_audit,
    mock_dcf,
    mock_forecast,
    mock_llm,
    mock_moat,
    mock_risk,
)


class TestAgentLeadSynthesizer(unittest.TestCase):
    """Code-assembled reports with LLM-written, code-verified prose."""

    def setUp(self):
        self.synthesizer = LeadSynthesizerAgent()
        resolver = patch("app.agents.shared.citations.resolve_chunk_metadata", return_value={})
        resolver.start()
        self.addCleanup(resolver.stop)

    def _synth(self, llm, **kwargs):
        defaults = dict(ticker="aapl", company_name="Apple Inc.", fiscal_year=2025)
        with patch.object(self.synthesizer, "_get_llm", return_value=llm):
            return self.synthesizer.synthesize(**{**defaults, **kwargs})

    def _full(self, llm, **kwargs):
        return self._synth(
            llm,
            business_moat=mock_moat(), financial_audit=mock_audit(), forecast=mock_forecast(),
            dcf_valuation=mock_dcf(), risk_audit=mock_risk(), **kwargs,
        )

    def test_01_system_prompt_renders(self):
        prompt = render_prompt("lead_synthesizer", "prompt_lead_synthesizer.j2", ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025)
        self.assertIn("Managing Director of Global Equity Research", prompt)
        self.assertIn("ResearchSynthesisPayload", prompt)
        self.assertIn("Prose only", prompt)

    def test_02_full_report_is_code_assembled_with_verbatim_tables(self):
        dcf, risk, forecast = mock_dcf(), mock_risk(), mock_forecast()
        report = self._full(mock_llm(FULL_PROSE))
        md = report.full_markdown_report

        self.assertEqual(report.ticker, "AAPL")
        self.assertEqual(report.synthesis_provenance, "llm_structured")
        self.assertEqual(report.implied_fair_value_per_share, dcf.implied_fair_value_per_share)
        self.assertEqual(report.valuation_stance, dcf.valuation_stance)
        self.assertIsInstance(report.three_pillar_thesis, ThreePillarThesis)
        self.assertIn("2.2B+ active devices", report.three_pillar_thesis.pillar_1_business_moat)

        for heading in (
            "# Institutional Equity Research Report: Apple Inc. (AAPL)",
            "## Executive Valuation Dashboard",
            "## 1. Executive Summary",
            "## 2. Institutional 3-Pillar Investment Thesis",
            "## 3. Business Operations & Economic Moat Analysis",
            "## 4. Audited Financial Statements & Quality of Earnings",
            f"## 5. {forecast.forecast_horizon_years}-Year Financial Forecast & Cash Flow Schedule",
            "## 6. Discounted Cash Flow (DCF) Valuation & Sensitivity",
            "## 7. Material Risk Factors & Existential Overhangs",
            "## 8. Regulatory Disclaimers & Audit Breadcrumbs",
        ):
            self.assertIn(heading, md)
        for table in (
            dcf.wacc_audit.formula_breakdown_markdown, dcf.sensitivity_matrix_markdown, dcf.equity_bridge_markdown,
            dcf.scenario_table_markdown, forecast.forecast_table_markdown, risk.risk_matrix_markdown,
            risk.evidence_table_markdown,
        ):
            self.assertIn(table.strip(), md)
        for warning in dcf.data_quality.warnings:
            self.assertIn(warning, md)
        self.assertIn(f"**${dcf.implied_fair_value_per_share:,.2f}**", md)
        self.assertIn("**Bottom line**: Overvalued on the base case", md)
        self.assertIn("> *Synthesis Provenance: `llm_structured`*", md)
        self.assertEqual(md.count("| R1 | Antitrust Scrutiny on App Store"), 1)

    def test_03_dcf_route_with_year_substitution(self):
        report = self._synth(
            mock_llm(FULL_PROSE), financial_audit=mock_audit(), forecast=mock_forecast(),
            dcf_valuation=mock_dcf(), year_substituted=True, query_type="dcf_valuation_only",
        )
        md = report.full_markdown_report
        self.assertTrue(report.year_substituted)
        self.assertIn("Filing Provenance Note", md)
        self.assertIn("# DCF Valuation & Intrinsic Sensitivity: Apple Inc. (AAPL)", md)
        self.assertIn("## 4. Equity Bridge & Scenario Valuation", md)
        self.assertNotIn("Business Operations", md)

    def test_04_untraceable_figures_and_layout_are_removed_from_prose(self):
        payload = FULL_PROSE.model_copy(update={
            "executive_summary": (
                "Revenue reached $416.2 billion in FY2025. Management will hit $999,999M of revenue by 2030. "
                "The Analysis Date is October 2023."
            ),
            "valuation_commentary": f"## Heading the model should not write\n| a | b |\nTerminal value is {TV_SHARE} of enterprise value.",
        })
        report = self._full(mock_llm(payload))
        md = report.full_markdown_report
        self.assertIn("Revenue reached $416.2 billion in FY2025.", md)
        self.assertNotIn("999,999", md)
        self.assertNotIn("October 2023", md)
        self.assertNotIn("Heading the model should not write", md)
        self.assertIn(f"Terminal value is {TV_SHARE} of enterprise value.", md)
        self.assertIn("1 sentence(s) of commentary were removed", md)

    def test_06_risk_prompt_embeds_verified_context(self):
        risk = mock_risk()
        prompt = render_prompt(
            "lead_synthesizer", "prompt_lead_synthesizer.j2", ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025,
            risk_audit=risk, query_type="risk_factors_only",
        )
        self.assertIn("VERIFIED RISK MATRIX (embed verbatim):", prompt)
        self.assertIn(risk.risk_matrix_markdown, prompt)
        self.assertIn("Primary Existential Threat: R1", prompt)
        self.assertIn('Evidence [narrative-chunk-item1a-01 | Item 1A]: "The Company is subject to complex', prompt)
        self.assertIn("YEAR-OVER-YEAR DISCLOSURE CHANGES (FY2025 vs FY2024)", prompt)
        self.assertIn("`bear_case_commentary`", prompt)
        self.assertIn("Do not state an analysis date", prompt)

    def test_07_template_default_report_is_complete(self):
        report = self._full(failing_llm())
        md = report.full_markdown_report
        self.assertEqual(report.synthesis_provenance, "template_default")
        self.assertIn("## 8. Regulatory Disclaimers & Audit Breadcrumbs", md)
        self.assertIn(mock_dcf().sensitivity_matrix_markdown.strip(), md)
        self.assertIn("Primary moat: High Switching Costs", report.three_pillar_thesis.pillar_1_business_moat)
        self.assertIn(FAIR, report.executive_summary)

    def test_08_risk_route_renders_matrix_once_and_details_in_code(self):
        risk = mock_risk()
        report = self._synth(mock_llm(FULL_PROSE), risk_audit=risk, query_type="risk_factors_only")
        md = report.full_markdown_report
        self.assertIn("# Material Risk Factors & Existential Overhangs: Apple Inc. (AAPL)", md)
        self.assertIn("**Overall Risk Profile**: **High** | **Severe Risks**: 2 of 3 | **Disclosure Baseline**: vs FY2024", md)
        self.assertEqual(md.count("| R1 | Antitrust Scrutiny on App Store"), 1)
        self.assertIn("### R1 — Antitrust Scrutiny on App Store", md)
        self.assertLess(md.index("## 2. Verified Risk Matrix"), md.index("| R1 | Antitrust Scrutiny"))

        moat_report = self._synth(
            mock_llm(FULL_PROSE), business_moat=mock_moat(), risk_audit=risk, query_type="business_moat_only",
        )
        self.assertNotIn("Verified Risk Matrix", moat_report.full_markdown_report)
        self.assertIn("## 2. Segments & Disaggregated Revenue Architecture", moat_report.full_markdown_report)

    def test_09_yoy_claims_stripped_without_baseline(self):
        payload = FULL_PROSE.model_copy(update={
            "risk_commentary": (
                "- Antitrust intervention is the primary existential threat.\n"
                "- A significant year-over-year change is the heightened focus on cybersecurity threats.\n"
                "- Supply concentration is unchanged in structure. Compared to the prior year, the language is broader."
            ),
            "executive_summary": "Antitrust is the primary threat. The year-over-year disclosures expanded on tariffs.",
        })
        no_baseline = mock_risk().model_copy(update={"disclosure_changes": None})
        report = self._synth(mock_llm(payload), risk_audit=no_baseline, query_type="risk_factors_only")
        md = report.full_markdown_report
        self.assertNotIn("heightened focus on cybersecurity", md)
        self.assertNotIn("Compared to the prior year", md)
        self.assertIn("- Supply concentration is unchanged in structure.", md)
        self.assertEqual(report.executive_summary, "Antitrust is the primary threat.")
        self.assertIn("year-over-year disclosure changes were not assessed", md)

        report = self._synth(mock_llm(payload), risk_audit=mock_risk(), query_type="risk_factors_only")
        self.assertIn("heightened focus on cybersecurity", report.full_markdown_report)

    def test_10_risk_audit_trail_rendered_in_code(self):
        base = mock_risk()
        risk = base.model_copy(update={"data_quality": base.data_quality.model_copy(update={
            "notes": ["1 of 3 quantified Item 1A disclosures are not cited by any verified risk."],
            "quantified_uncovered": ["Sales to one direct customer represented 22% of total revenue."],
        })})
        md = self._synth(mock_llm(FULL_PROSE), risk_audit=risk, query_type="risk_factors_only").full_markdown_report
        self.assertIn("**Risk analysis data-quality notes**\n- 1 of 3 quantified Item 1A disclosures", md)
        self.assertIn('- "Sales to one direct customer represented 22% of total revenue."', md)
        self.assertLess(md.index("## 6. Filing Evidence & Audit Trail"), md.index("**Risk analysis data-quality notes**"))
        self.assertEqual(md.count("Synthesis Provenance"), 1)

    def test_11_missing_sections_are_disclosed_with_pipeline_notes(self):
        note = "Risk Analyst failed (TimeoutError: read timed out); its section is omitted from this report."
        report = self._synth(
            mock_llm(FULL_PROSE), business_moat=mock_moat(), financial_audit=mock_audit(),
            forecast=mock_forecast(), dcf_valuation=mock_dcf(), pipeline_notes=[note, note],
        )
        md = report.full_markdown_report
        self.assertIn("_The risk factor analysis is unavailable for this report.", md)
        self.assertEqual(md.count("- " + note), 1)

    def test_12_single_topic_route_without_its_artifact_uses_full_layout(self):
        report = self._synth(
            failing_llm(), financial_audit=mock_audit(), query_type="risk_factors_only",
            pipeline_notes=["Risk Analyst failed (ValueError: bad output); its section is omitted from this report."],
        )
        md = report.full_markdown_report
        self.assertIn("# Institutional Equity Research Report: Apple Inc. (AAPL)", md)
        self.assertIn("_The risk factor analysis is unavailable", md)
        self.assertIsNone(report.implied_fair_value_per_share)
        self.assertIsNone(report.valuation_stance)

    def test_13_financial_audit_route_embeds_statements(self):
        md = self._synth(
            mock_llm(FULL_PROSE), financial_audit=mock_audit(), query_type="financial_audit_only",
        ).full_markdown_report
        self.assertIn("# Financial Statement Audit & Quality of Earnings: Apple Inc. (AAPL)", md)
        self.assertIn("| Total net sales | $383,285 | $391,035 | $416,161 |", md)
        self.assertIn("## 9. Audited Statements", md)

    def test_14_none_guard_formatting(self):
        md = self._synth(failing_llm(), financial_audit=mock_audit(), query_type="financial_audit_only").full_markdown_report
        self.assertIn("| Stock-based comp. | N/A | N/A | N/A |", md)


if __name__ == "__main__":
    unittest.main()
