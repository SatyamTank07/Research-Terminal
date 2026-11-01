"""Item 1 location, the search fallback, the filing load and the coverage notes the Strategist reads before judging."""

import unittest
from unittest.mock import MagicMock, patch

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.shared.filing.sections import build_section_context
from app.agents.specialized.business_strategist.context import (
    ITEM1_CHAR_BUDGET,
    RAG_FALLBACK_QUERIES,
    filing_items,
    load_filing,
    locate_item1,
    pool_chunk,
    read_item1,
)
from app.agents.tools.business_disclosure_tools import FilingChunk

CONTEXT = "app.agents.specialized.business_strategist.context"


def _chunk(chunk_id: str, index: int, content: str = "Our marketplace connects buyers and sellers.") -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=chunk_id, document_id="doc", ticker="TEST", fiscal_year=2025, item="Item 1",
        breadcrumb=f"TEST > Item 1 ({chunk_id})", sub_section="Business", content=content, chunk_index=index,
    )


class TestLocateAndLoad(unittest.TestCase):
    def test_locate_failure_returns_not_found(self):
        with patch(f"{CONTEXT}.locate_section_chunks", side_effect=RuntimeError("db down")):
            self.assertEqual(locate_item1("TEST", 2025), ([], "not_found"))

    def test_load_failure_returns_a_note(self):
        with patch(f"{CONTEXT}.load_filing_chunks", side_effect=RuntimeError("db down")):
            filing, notes = load_filing("TEST", 2025)
        self.assertEqual(filing, [])
        self.assertEqual(notes, ["Filing tables could not be loaded; segment and revenue-line figures not extracted."])

    def test_items_are_inferred_only_when_labels_were_wrong(self):
        filing = [
            FilingChunk(chunk_id="h1", chunk_index=0, chunk_type="table", item="Cover Page", content="| Item 1. | Business |"),
            FilingChunk(chunk_id="b1", chunk_index=1, chunk_type="narrative", item="Cover Page", content="We sell."),
        ]
        self.assertEqual(filing_items(filing, "item_label"), {})
        self.assertEqual(filing_items(filing, "heading_boundary"), {"h1": "Item 1", "b1": "Item 1"})


class TestReadItem1(unittest.TestCase):
    def test_full_section_from_label(self):
        chunks = [_chunk("i1", 0), _chunk("i2", 1)]
        text = read_item1("TEST", 2025, chunks, "item_label")
        self.assertEqual(text.coverage_mode, "full_section")
        self.assertEqual(text.item1_chunks, 2)
        self.assertEqual(list(text.pool), ["i1", "i2"])
        self.assertEqual((text.context, text.chars_read), build_section_context(chunks, ITEM1_CHAR_BUDGET)[::2])
        self.assertEqual(text.notes, ())

    def test_heading_boundary_is_noted(self):
        text = read_item1("TEST", 2025, [_chunk("i1", 0)], "heading_boundary")
        self.assertIn("section headings", text.notes[0])

    def test_search_fallback_dedupes_and_orders_by_filing_position(self):
        rag = MagicMock(side_effect=[[_chunk("b", 5), _chunk("a", 2)], [_chunk("a", 2)], RuntimeError("timeout")])
        with patch(f"{CONTEXT}.retrieve_10k_narrative", rag):
            text = read_item1("TEST", 2025, [], "not_found")
        self.assertEqual(rag.call_count, len(RAG_FALLBACK_QUERIES))
        self.assertEqual(list(text.pool), ["a", "b"])
        self.assertEqual(text.coverage_mode, "rag_fallback")
        self.assertEqual(text.item1_chunks, 0)
        self.assertIn("could not be located", text.notes[0])

    def test_no_text_at_all(self):
        with patch(f"{CONTEXT}.retrieve_10k_narrative", return_value=[]):
            text = read_item1("TEST", 2025, [], "not_found")
        self.assertEqual((text.context, text.coverage_mode, text.chars_read), ("", "no_context", 0))
        self.assertEqual(text.notes, ("No Item 1 text could be retrieved for TEST FY2025.",))

    def test_pool_chunk_prefers_the_inferred_item(self):
        chunk = FilingChunk(chunk_id="m1", chunk_index=9, chunk_type="narrative", item="Cover Page", content="x")
        self.assertEqual(pool_chunk(chunk, "TEST", 2025, {"m1": "Item 7"})["item"], "Item 7")
        self.assertEqual(pool_chunk(chunk, "TEST", 2025, {})["item"], "Cover Page")


if __name__ == "__main__":
    unittest.main()
