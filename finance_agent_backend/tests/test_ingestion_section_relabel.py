"""Relabel tests: re-deriving section labels of stored chunks without re-parsing or re-embedding.

Validates:
1. Table-rendered headings stored as chunks relabel a Cover Page / Item 8 filing into its real items,
   including the cross-reference artifact and the PART boundary the old parser produced.
2. A correctly labelled filing is left untouched.
3. A cross-reference artifact inside a part continues the running section; one that opened a new
   part takes the only standard item that fits; an ambiguous one is left as is with a warning.
4. Applying a plan and planning again changes nothing (idempotent).
"""

import unittest
from dataclasses import replace
from typing import List, Optional, Tuple

from app.services.ingestion.chunker import SECChunker
from app.services.ingestion.sec_parser import ParsedSection, SECFilingMetadata
from app.services.ingestion.section_relabel import (
    StoredChunk,
    derive_section_labels,
    format_plan,
    heading_from_table_chunk,
    plan_relabel,
)

META = SECFilingMetadata(
    ticker="EXCO", company_name="Example Holdings, Inc.", filing_type="10-K", fiscal_year=2031,
    source_filename="exco-20311231.htm",
)
TEXT = "Narrative paragraph. " * 12


def _heading(number: str, title: str) -> str:
    return f"| Item {number}. | {title} |\n| :--- | :--- |"


class _Builder:
    """Builds stored chunks the way the old parser + chunker labelled them."""

    def __init__(self):
        self.chunks: List[StoredChunk] = []

    def add(self, part: Optional[str], item: str, title: str, content: str = TEXT, chunk_type: str = "narrative"):
        self.chunks.append(StoredChunk(
            chunk_index=len(self.chunks), chunk_type=chunk_type, content=content,
            part=part, item=item, section_title=title,
        ))
        return self

    def table(self, part: Optional[str], item: str, title: str, content: str):
        return self.add(part, item, title, content, chunk_type="table")


def _rows(chunks: List[StoredChunk], meta: SECFilingMetadata = META) -> List[Tuple[str, StoredChunk, str]]:
    chunker = SECChunker()
    return [
        (f"id-{c.chunk_index}", c, chunker.build_breadcrumb(meta, ParsedSection(c.part, c.item, c.section_title or "")))
        for c in chunks
    ]


def _apply(chunks: List[StoredChunk], plan) -> List[StoredChunk]:
    after = {change.chunk_index: change.after for change in plan.changes}
    return [
        replace(c, part=after[c.chunk_index].part, item=after[c.chunk_index].item, section_title=after[c.chunk_index].section_title)
        if c.chunk_index in after else c
        for c in chunks
    ]


def _table_heading_filing() -> List[StoredChunk]:
    """Old labels of a filing whose item headings are one-row tables."""
    b = _Builder()
    b.add(None, "Cover Page", "Document Overview")
    b.table(None, "Cover Page", "Document Overview", "| Page |\n| :--- |\n| Item 1. | Business | 3 |\n| Item 1A. | Risk Factors | 6 |")
    b.table("PART I", "Cover Page", "PART I Overview", _heading("1", "Business"))
    b.add("PART I", "Cover Page", "PART I Overview")
    b.table("PART I", "Cover Page", "PART I Overview", _heading("1A", "Risk Factors"))
    b.add("PART I", "Cover Page", "PART I Overview")
    b.add("PART I", "Cover Page", "PART I Overview")
    b.table("PART I", "Cover Page", "PART I Overview", _heading("3", "Legal Proceedings"))
    # A short "... see Item 8 of Part II ..." paragraph was taken as a heading here.
    b.table("PART I", "Item 8", "of Part II, “Financial Statements and Supplementary Data — Note 7", _heading("4", "Mine Safety Disclosures"))
    b.table("PART II", "Item 8", "PART II Overview", _heading("5", "Market for the Registrant’s Common Stock"))
    b.add("PART II", "Item 8", "PART II Overview")
    b.table("PART II", "Item 8", "PART II Overview", _heading("7", "Management’s Discussion and Analysis"))
    b.add("PART II", "Item 8", "PART II Overview")
    b.table("PART II", "Item 8", "PART II Overview", _heading("8", "Financial Statements and Supplementary Data"))
    b.table("PART II", "Item 8", "PART II Overview", "| Line item | 2030 | 2031 |\n| :--- | ---: | ---: |\n| Net sales | 100 | 120 |")
    return b.chunks


class TestSectionRelabel(unittest.TestCase):

    def test_01_heading_table_detection(self):
        self.assertEqual(heading_from_table_chunk(_heading("1A", "Risk Factors")), ("item", "Item 1A", "Risk Factors"))
        self.assertIsNone(heading_from_table_chunk("| Item 1A. | Risk Factors | 6 |\n| :--- | :--- | :--- |"))
        self.assertIsNone(heading_from_table_chunk("| Line item | 2031 |\n| :--- | ---: |\n| Net sales | 120 |"))

    def test_02_table_heading_filing_relabelled(self):
        labels, warnings = derive_section_labels(_table_heading_filing())

        self.assertEqual(warnings, [])
        self.assertEqual(
            [(label.part, label.item) for label in labels],
            [
                (None, "Cover Page"), (None, "Cover Page"),
                ("PART I", "Item 1"), ("PART I", "Item 1"),
                ("PART I", "Item 1A"), ("PART I", "Item 1A"), ("PART I", "Item 1A"),
                ("PART I", "Item 3"), ("PART I", "Item 4"),
                ("PART II", "Item 5"), ("PART II", "Item 5"),
                ("PART II", "Item 7"), ("PART II", "Item 7"),
                ("PART II", "Item 8"), ("PART II", "Item 8"),
            ],
        )
        self.assertEqual(labels[5].title, "Risk Factors")

    def test_03_plan_rebuilds_breadcrumb_and_sub_section(self):
        chunks = _table_heading_filing()
        plan = plan_relabel(META, _rows(chunks))
        by_index = {change.chunk_index: change.after for change in plan.changes}

        self.assertEqual(
            by_index[5].breadcrumb,
            "Example Holdings, Inc. (EXCO) > 10-K FY2031 > PART I > Item 1A: Risk Factors",
        )
        self.assertEqual(by_index[5].sub_section, "Risk Factors")
        self.assertNotIn(0, by_index)  # cover page chunks keep their labels
        self.assertEqual(plan.before_counts["Cover Page"], 8)
        self.assertEqual(plan.after_counts["Item 1A"], 3)
        self.assertIn("Item 1A", format_plan(plan))

    def test_04_correctly_labelled_filing_unchanged(self):
        b = _Builder()
        b.add(None, "Cover Page", "Document Overview")
        b.table(None, "Cover Page", "Document Overview", "| Page |\n| :--- |\n| Item 1. | Business | 3 |")
        b.add("PART I", "Item 1", "Business").add("PART I", "Item 1", "Business")
        b.add("PART I", "Item 1A", "Risk Factors")
        b.add("PART II", "Item 7", "Management’s Discussion and Analysis")
        b.table("PART II", "Item 7", "Management’s Discussion and Analysis", "| Line item | 2031 |\n| :--- | ---: |\n| Revenue | 1 |")
        b.add("PART II", "Item 8", "Financial Statements and Supplementary Data")
        b.add("PART III", "Item 9C", "PART III Overview")  # part preamble keeps the running item
        b.add("PART III", "Item 10", "Directors, Executive Officers and Corporate Governance")

        plan = plan_relabel(META, _rows(b.chunks))
        self.assertEqual(plan.changes, [])
        self.assertEqual(plan.warnings, [])

    def test_05_artifact_inside_part_continues_section(self):
        b = _Builder()
        b.add("PART II", "Item 7", "Management’s Discussion and Analysis")
        b.add("PART II", "Item 15", "of this Annual Report on Form 10-K for further discussion")
        b.add("PART II", "Item 15", "of this Annual Report on Form 10-K for further discussion")
        b.add("PART II", "Item 7A", "Quantitative and Qualitative Disclosures About Market Risk")

        labels, warnings = derive_section_labels(b.chunks)
        self.assertEqual([label.item for label in labels], ["Item 7", "Item 7", "Item 7", "Item 7A"])
        self.assertEqual(labels[1].title, "Management’s Discussion and Analysis")
        self.assertEqual(warnings, [])

    def test_06_artifact_opening_a_part_takes_the_only_fitting_item(self):
        b = _Builder()
        b.add("PART III", "Item 10", "Directors, Executive Officers and Corporate Governance")
        b.add("PART IV", "Item 8", "of this report)")
        b.add("PART IV", "Item 16", "Form 10-K Summary")

        labels, warnings = derive_section_labels(b.chunks)
        self.assertEqual([(label.part, label.item) for label in labels], [("PART III", "Item 10"), ("PART IV", "Item 15"), ("PART IV", "Item 16")])
        self.assertEqual(labels[1].title, "Exhibits and Financial Statement Schedules")
        self.assertEqual(warnings, [])

    def test_07_ambiguous_artifact_left_unchanged_with_warning(self):
        b = _Builder()
        b.add("PART I", "Item 1", "Business")
        b.add("PART II", "Item 3", "of Part I")  # Part II, nothing after it: items 5..9C all fit
        with self.assertLogs("finance_agent.ingestion.section_relabel", level="WARNING"):
            labels, warnings = derive_section_labels(b.chunks)
        self.assertEqual(labels[1].item, "Item 3")
        self.assertEqual(len(warnings), 1)

    def test_08_part_boundary_after_artifact_keeps_running_item(self):
        """An artifact's item carried across a PART boundary is not trusted; a real heading's is."""
        b = _Builder()
        b.add("PART II", "Item 9B", "Other Information")
        b.add("PART III", "Item 9C", "PART III Overview")  # Item 9C's own text was too short to store
        labels, _ = derive_section_labels(b.chunks)
        self.assertEqual(labels[1].item, "Item 9C")

        labels, _ = derive_section_labels(_table_heading_filing())
        self.assertEqual(labels[9].item, "Item 5")
        self.assertEqual(labels[10].item, "Item 5")

    def test_09_idempotent(self):
        chunks = _table_heading_filing()
        first = plan_relabel(META, _rows(chunks))
        relabelled = _apply(chunks, first)

        rows = _rows(relabelled)
        second = plan_relabel(META, rows)
        self.assertGreater(len(first.changes), 0)
        self.assertEqual(second.changes, [])

    def test_10_non_10k_artifacts_are_not_inferred(self):
        b = _Builder()
        b.add("PART I", "Item 2", "Management’s Discussion")
        b.add("PART II", "Item 1", "of Part I")
        meta = SECFilingMetadata(ticker="EXCO", company_name="Example Holdings, Inc.", filing_type="10-Q", fiscal_year=2031)
        plan = plan_relabel(meta, _rows(b.chunks, meta))
        self.assertEqual(plan.changes, [])
        self.assertEqual(len(plan.warnings), 1)


if __name__ == "__main__":
    unittest.main()
