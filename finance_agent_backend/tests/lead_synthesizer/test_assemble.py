"""Assembly rules: citations never break the report, and a report without filing sources says so."""

import unittest
from unittest.mock import MagicMock, patch

from app.agents.shared.citations import CitationRegistry, SourceRef, marker
from app.agents.specialized.lead_synthesizer.assemble import cite_prose, finalize_citations
from app.agents.specialized.lead_synthesizer.report_citations import (
    CITATIONS_FAILED_NOTE,
    EVIDENCE_NOTE,
    NO_CITATIONS_NOTE,
    NullCiter,
    ReportCiter,
)

REPORT = f"Revenue rose{marker(1)}.\n\n{EVIDENCE_NOTE}\n"


def _citer(finalize) -> ReportCiter:
    citer = ReportCiter(CitationRegistry(resolver=MagicMock(return_value={})), 2025)
    citer.registry.finalize = finalize
    return citer


class TestFinalizeCitations(unittest.TestCase):
    def test_a_finalize_failure_strips_the_markers_and_says_so(self):
        def boom(markdown):
            raise RuntimeError("db down")

        with self.assertLogs("finance_agent.agents.lead_synthesizer", "ERROR"):
            markdown, sources = finalize_citations(_citer(boom), REPORT)
        self.assertEqual(sources, [])
        self.assertNotIn(marker(1), markdown)
        self.assertNotIn(EVIDENCE_NOTE, markdown)
        self.assertIn(NO_CITATIONS_NOTE, markdown)
        self.assertTrue(markdown.endswith(CITATIONS_FAILED_NOTE + "\n"))

    def test_a_report_citing_only_market_data_swaps_in_the_no_citations_note(self):
        market = SourceRef(n=1, kind="market", label="Share price", provider="yfinance")
        markdown, sources = finalize_citations(_citer(lambda md: (md, [market])), REPORT)
        self.assertEqual(sources, [market])
        self.assertNotIn(EVIDENCE_NOTE, markdown)
        self.assertIn(NO_CITATIONS_NOTE, markdown)

    def test_a_report_built_without_a_citer_carries_no_markers(self):
        markdown, sources = finalize_citations(NullCiter(), REPORT)
        self.assertEqual((markdown, sources), (f"Revenue rose.\n\n{EVIDENCE_NOTE}\n", []))


class TestCiteProse(unittest.TestCase):
    def test_a_citation_build_failure_publishes_uncited_prose_with_a_note(self):
        notes = []
        prose = {"executive_summary": "Revenue was $1.0M."}
        target = "app.agents.specialized.lead_synthesizer.report_citations.ReportCiter.figure_entries"
        with patch(target, side_effect=RuntimeError("bad provenance")), \
                self.assertLogs("finance_agent.agents.lead_synthesizer", "ERROR"):
            citer, cited = cite_prose(prose, 2025, {}, notes)
        self.assertIsInstance(citer, NullCiter)
        self.assertEqual(cited, prose)
        self.assertEqual(notes, ["Filing citations could not be built for this report."])
