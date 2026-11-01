"""The Strategist's output from verified module results: tally order, citations and the not-assessed reason."""

import unittest

from app.agents.specialized.business_strategist.assemble import (
    assemble_strategy,
    build_citations,
    not_assessed_reason,
    record_tally,
)
from app.agents.specialized.business_strategist.context import FilingChunk
from app.agents.specialized.business_strategist.evidence import StrategyEvidence, VerifyTally
from app.agents.specialized.business_strategist.modules.business_model import BusinessModel
from app.agents.specialized.business_strategist.modules.competition_and_moat import not_assessed_competition
from app.agents.specialized.business_strategist.modules.market_power import MarketPower
from app.agents.specialized.business_strategist.modules.revenue_mix import extract_segment_disclosures
from app.agents.specialized.business_strategist.state_business_strategist import StrategistDataQuality
from tests.business_strategist.filing_fixtures import filing


def _evidence(chunk_id, claim="pricing_power", subject=None):
    return StrategyEvidence(claim=claim, subject=subject, chunk_id=chunk_id, quote="q", item="Item 7", breadcrumb="MD&A")


class TestRecord(unittest.TestCase):
    def test_tallies_add_up_and_notes_append(self):
        dq = StrategistDataQuality(notes=["context note"])
        record_tally(VerifyTally(claims_proposed=2, quotes_rejected=1, notes=["a"]), dq)
        record_tally(VerifyTally(claims_proposed=1, quotes_off_topic=3, notes=["b"]), dq)
        self.assertEqual((dq.claims_proposed, dq.quotes_rejected, dq.quotes_off_topic), (3, 1, 3))
        self.assertEqual(dq.notes, ["context note", "a", "b"])

    def test_not_assessed_reason(self):
        self.assertEqual(not_assessed_reason("failed"), "the qualitative LLM call failed")
        self.assertEqual(not_assessed_reason("malformed"), "the model returned malformed output")
        self.assertEqual(not_assessed_reason("skipped"), "no filing text was available")


class TestCitations(unittest.TestCase):
    def test_evidence_chunks_first_then_tables_once_each(self):
        chunks = filing()
        extraction = extract_segment_disclosures(chunks, 2025)
        pool = {"n2": {"chunk_id": "n2", "item": "Item 8", "breadcrumb": "Note 10"}}
        evidence = [_evidence("n2"), _evidence("n2", claim="customer_concentration"), _evidence("gone")]
        citations = build_citations(evidence, pool, extraction, chunks, {"t3": "Item 8"}, "TEST", 2025)
        self.assertEqual([c["chunk_id"] for c in citations], ["n2", "gone", "t3", "t5"])
        self.assertEqual(citations[1], {"chunk_id": "gone", "ticker": "TEST", "fiscal_year": 2025,
                                        "item": "Item 7", "breadcrumb": "MD&A"})


class TestAssemble(unittest.TestCase):
    def test_module_results_become_the_output_in_verify_order(self):
        chunks = filing()
        extraction = extract_segment_disclosures(chunks, 2025)
        competition = not_assessed_competition("the qualitative LLM call failed")
        competition.tally.notes.append("moat note")
        market = MarketPower("Not assessed: x.", "Not explicitly disclosed in 10-K", [_evidence("t1")],
                             VerifyTally(notes=["market note"]))
        business = BusinessModel("Summary.", "Architecture.", [], [], VerifyTally(notes=["business note"]))
        dq = StrategistDataQuality()
        out = assemble_strategy("TEST", 2025, extraction, competition, market, business, {}, chunks, {}, dq)
        self.assertEqual(out.data_quality.notes, ["moat note", "market note", "business note"])
        self.assertEqual(out.reportable_segments, ["Retail", "Cloud"])
        self.assertEqual((out.moat_assessment, out.business_summary), ("not_assessed", "Summary."))
        self.assertEqual(out.data_quality.claims_verified, 1)
        self.assertEqual(out.citations[0]["chunk_id"], "t1")
        self.assertIsInstance(chunks[0], FilingChunk)


if __name__ == "__main__":
    unittest.main()
