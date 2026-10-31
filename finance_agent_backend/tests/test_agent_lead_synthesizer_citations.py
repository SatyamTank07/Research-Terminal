"""Cited research reports: inline markers <-> sources (lead synthesizer + report_citations)."""

import re
import unittest
from unittest.mock import patch

from app.agents.specialized.lead_synthesizer.agent_lead_synthesizer import LeadSynthesizerAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.shared.citations import SourceChunk
from tests.test_agent_lead_synthesizer import (
    FULL_PROSE,
    _create_mock_dcf,
    _create_mock_forecast,
    _create_mock_risk,
    _mock_llm,
)
from tests.test_report_citations import audit_with_provenance, moat_with_evidence

MARKER = re.compile(r"\[\[(\d+)\]\]\(#cite-\1\)")


def _fake_resolver(ids):
    return {
        cid: SourceChunk(chunk_id=cid, document_id="doc-aapl-2025", chunk_index=i, ticker="AAPL", fiscal_year=2025,
                         item="Item 8", breadcrumb=f"Apple Inc. (AAPL) > Item 8 > {cid}")
        for i, cid in enumerate(ids)
    }


class TestSynthesizerCitations(unittest.TestCase):
    def setUp(self):
        self.synth = LeadSynthesizerAgent()
        patcher = patch("app.agents.shared.citations.resolve_chunk_metadata", side_effect=_fake_resolver)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _report(self, **overrides):
        kwargs = dict(
            business_moat=moat_with_evidence(), financial_audit=audit_with_provenance(),
            forecast=_create_mock_forecast(), dcf_valuation=_create_mock_dcf(), risk_audit=_create_mock_risk(),
        )
        kwargs.update(overrides)
        with patch.object(self.synth, "_get_llm", return_value=_mock_llm(FULL_PROSE)):
            return self.synth.synthesize(ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025, **kwargs)

    def test_markers_and_sources_are_a_bijection(self):
        report = self._report()
        numbers = {int(n) for n in MARKER.findall(report.full_markdown_report)}
        self.assertTrue(numbers)
        self.assertEqual(numbers, set(range(1, len(report.sources) + 1)))
        self.assertEqual([s.n for s in report.sources], list(range(1, len(report.sources) + 1)))
        self.assertEqual({s.kind for s in report.sources}, {"verbatim", "derived", "market"})

    def test_prose_figures_are_attributed(self):
        md = self._report().full_markdown_report
        self.assertRegex(md, r"free cash flow was \$98,767M(\[\[\d+\]\]\(#cite-\d+\))+\.")

    def test_verbatim_quotes_become_sources(self):
        report = self._report()
        quotes = {s.quote for s in report.sources if s.kind == "verbatim"}
        self.assertIn("The Company is subject to complex and changing laws and regulations worldwide", quotes)
        self.assertIn("The Company faces substantial competition from Android OEMs", quotes)
        verbatim = next(s for s in report.sources if s.kind == "verbatim")
        self.assertEqual(verbatim.document_id, "doc-aapl-2025")
        self.assertEqual(len(verbatim.chunks), 1)

    def test_structured_fields_and_prompt_context_carry_no_markers(self):
        report = self._report()
        for text in (
            report.executive_summary,
            report.three_pillar_thesis.pillar_1_business_moat,
            report.three_pillar_thesis.pillar_2_financial_durability,
            report.three_pillar_thesis.pillar_3_valuation_asymmetry,
        ):
            self.assertNotIn("#cite-", text)
        context = render_prompt(
            "lead_synthesizer", ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025,
            query_type="full_10k_report", financial_audit=audit_with_provenance(), business_moat=moat_with_evidence(),
            risk_audit=_create_mock_risk(), forecast=_create_mock_forecast(), dcf_valuation=_create_mock_dcf(),
        )
        self.assertNotIn("#cite-", context)

    def test_citation_failure_publishes_report_without_markers(self):
        target = "app.agents.specialized.lead_synthesizer.agent_lead_synthesizer.ReportCiter.figure_entries"
        with patch(target, side_effect=RuntimeError("boom")):
            report = self._report()
        self.assertNotIn("#cite-", report.full_markdown_report)
        self.assertEqual(report.sources, [])
        self.assertIn("Filing citations could not be built", report.full_markdown_report)

    def test_audit_rows_on_the_same_statement_share_a_marker(self):
        md = self._report(query_type="financial_audit_only").full_markdown_report
        found = re.search(r"\| Revenue(\[\[\d+\]\]\(#cite-\d+\)) \|", md)
        self.assertIsNotNone(found)
        self.assertIn(f"| Operating income{found.group(1)} |", md)
        self.assertRegex(md, r"\| Cash & equivalents\[\[\d+\]\]\(#cite-\d+\) \|")

    def test_computed_valuation_figures_get_no_misattributed_marker(self):
        # Rounded computed figures that, before unit-aware attribution, collided with unrelated statement rows.
        dcf, forecast = _create_mock_dcf(), _create_mock_forecast()
        sentence = (f"Terminal growth is about {dcf.terminal_growth_rate * 100:.0f}% while revenue compounds "
                    f"at roughly {forecast.revenue_cagr_pct:.0f}%.")
        prose = FULL_PROSE.model_copy(update={"valuation_commentary": sentence})
        with patch.object(self.synth, "_get_llm", return_value=_mock_llm(prose)):
            md = self.synth.synthesize(
                ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025, business_moat=moat_with_evidence(),
                financial_audit=audit_with_provenance(), forecast=_create_mock_forecast(), dcf_valuation=dcf,
                risk_audit=_create_mock_risk(),
            ).full_markdown_report
        self.assertIn(sentence, md)

    def test_lookup_failure_publishes_stripped_report_with_note(self):
        with patch("app.agents.shared.citations.resolve_chunk_metadata", side_effect=RuntimeError("db down")):
            report = self._report(query_type="business_moat_only")
        md = report.full_markdown_report
        self.assertNotIn("#cite-", md)
        self.assertEqual(report.sources, [])
        self.assertIn("Filing citations could not be built for this report.", md)
        self.assertNotIn("Every numbered marker in this report opens", md)

    def test_evidence_note_shown_when_citations_resolve(self):
        md = self._report(query_type="business_moat_only").full_markdown_report
        self.assertIn("Every numbered marker in this report opens the cited passage of the 10-K", md)

    def test_risk_detail_lines_keep_their_line_breaks(self):
        md = self._report().full_markdown_report
        self.assertIn("- **Filing-stated mitigation**: None stated in the filing\n- **Model lines hit**", md)

    def test_strategist_and_risk_lines_are_marked(self):
        md = self._report().full_markdown_report
        self.assertRegex(md, r"- Android OEMs\[\[\d+\]\]\(#cite-\d+\)")
        self.assertRegex(md, r"\*\*Disclosure\*\*: Global regulatory enforcement[^\n]*\[\[\d+\]\]\(#cite-\d+\)")
        self.assertRegex(md, r"\| Current Share Price\[\[\d+\]\]\(#cite-\d+\) \|")
