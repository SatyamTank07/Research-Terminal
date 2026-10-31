"""Locating a 10-K section's chunks (stored label, else the filing's own headings) and budgeting them
into prompt context. No database or network access: the filing rows are stubbed."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.shared.filing.sections import build_section_context, heading_item, locate_section_chunks

CONCENTRATION = (
    "We receive a significant amount of our revenue from a limited number of customers. "
    "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
    "another direct customer represented 14% of total revenue, all of which were primarily attributable "
    "to the Compute & Networking segment."
)


def _chunk(chunk_id: str, content: str, idx: int = 0, item: str = "Item 1A", fy: int = 2026) -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        ticker="TEST",
        fiscal_year=fy,
        item=item,
        breadcrumb=f"TEST > 10-K FY{fy} > PART I > {item}: Risk Factors",
        sub_section="Risk Factors",
        content=content,
        chunk_index=idx,
    )


def _row(idx: int, content: str, chunk_type: str = "narrative", item: str = "Cover Page"):
    chunk = SimpleNamespace(id=f"k{idx}", item=item, breadcrumb=f"X > {item}", sub_section=None,
                            content=content, chunk_index=idx, chunk_type=chunk_type)
    doc = SimpleNamespace(id="doc-1", ticker="TEST", fiscal_year=2026)
    return chunk, doc


class TestSectionLocation(unittest.TestCase):
    TOC = ("### Page | PART I | | Item 1. | Business | 3 | | Item 1A. | Risk Factors | 6 | "
           "| Item 1B. | Unresolved Staff Comments | 16 | | Item 2. | Properties | 17 |")
    LONG_BODY = "Our business depends on a small number of suppliers and customers in several regions. " * 6

    def test_heading_detection(self):
        self.assertEqual(heading_item("| Item 1A. | Risk Factors |\n| :--- | :--- |"), "1A")
        self.assertEqual(heading_item("| Item 6. | Reserved |"), "6")
        self.assertEqual(heading_item("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"), "7")
        self.assertIsNone(heading_item(self.TOC))
        self.assertEqual(heading_item("Item 1A. Risk Factors " + self.LONG_BODY), "1A")
        self.assertIsNone(heading_item("Item 1A of Part I describes factors that " + self.LONG_BODY))
        self.assertIsNone(heading_item("For more information, see Item 1A. Risk Factors."))
        self.assertIsNone(heading_item("| Item 99. | Nothing |"))

    def _locate(self, rows, labeled=None):
        query = MagicMock()
        query.order_by.return_value.all.return_value = rows
        with patch("app.agents.shared.filing.sections.fetch_section_chunks", return_value=labeled or []), \
                patch("app.agents.shared.filing.sections._filing_chunks_query", return_value=query):
            return locate_section_chunks("TEST", 2026, "Item 1A", db=MagicMock())

    def test_boundary_location_when_labels_are_wrong(self):
        rows = [
            _row(0, self.TOC, "table"),
            _row(1, "| Item 1. | Business |", "table"),
            _row(2, "We design products. " + self.LONG_BODY),
            _row(3, "| Item 1A. | Risk Factors |\n| :--- | :--- |", "table"),
            _row(4, "Risk one narrative. " + self.LONG_BODY),
            _row(5, "| Segment | Share |\n| :--- | :--- |\n| A | 40% |", "table"),
            _row(6, "Risk two narrative; see Item 7 for details. " + self.LONG_BODY),
            _row(7, "| Item 1B. | Unresolved Staff Comments |", "table"),
            _row(8, "None."),
            _row(9, "| Item 1C. | Cybersecurity |", "table"),
        ]
        chunks, located_by = self._locate(rows)
        self.assertEqual(located_by, "heading_boundary")
        self.assertEqual([c.chunk_index for c in chunks], [4, 6])
        self.assertTrue(all(c.item == "Item 1A" for c in chunks))

    def test_heading_inside_long_chunk_and_label_precedence(self):
        rows = [_row(0, "Item 1A. Risk Factors " + self.LONG_BODY), _row(1, "More risk text. " + self.LONG_BODY),
                _row(2, "Item 2. Properties " + self.LONG_BODY)]
        chunks, located_by = self._locate(rows)
        self.assertEqual(([c.chunk_index for c in chunks], located_by), ([0, 1], "heading_boundary"))

        labeled = [_chunk("L1", "Labeled risk text " + self.LONG_BODY)]
        chunks, located_by = self._locate(rows, labeled=labeled)
        self.assertEqual((chunks, located_by), (labeled, "item_label"))

        self.assertEqual(self._locate([_row(0, self.LONG_BODY)]), ([], "not_found"))

    def test_section_queries_are_scoped_to_annual_reports(self):
        from app.database import SessionLocal
        from app.agents.shared.filing.sections import _filing_chunks_query
        db = SessionLocal()
        try:
            sql = str(_filing_chunks_query(db, "TEST", 2026).statement)
        finally:
            db.close()
        self.assertIn("documents.filing_type", sql)


class TestSectionContext(unittest.TestCase):
    def test_full_section_within_budget(self):
        chunks = [_chunk("c1", "Summary bullet text " * 10, 0), _chunk("c2", CONCENTRATION, 1)]
        context, mode, chars = build_section_context(chunks, char_budget=10_000)
        self.assertEqual(mode, "full_section")
        self.assertIn("[chunk:c1]", context)
        self.assertIn("[chunk:c2] We receive a significant amount", context)
        self.assertGreater(chars, len(CONCENTRATION))

    def test_budgeted_section_keeps_summary_and_truncates_rest(self):
        summary = "Summary of Risk Factors • Risk one is here. • Risk two is here. • Risk three is here. " * 5
        body = "Detailed risk heading sentence. " + ("Long detailed paragraph text. " * 200)
        chunks = [_chunk("s1", summary, 0)] + [_chunk(f"b{i}", body, i + 1) for i in range(5)]
        context, mode, _ = build_section_context(chunks, char_budget=5_000)
        self.assertEqual(mode, "budgeted_section")
        self.assertIn(summary.strip(), context)
        self.assertIn("[chunk:b0] Detailed risk heading sentence.", context)
        self.assertIn("[...]", context)
        self.assertLessEqual(len(context), 5_000 + 200)
