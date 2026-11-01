"""Item 1A context: a filing that provides no risk factors, the prior-year pick, coverage notes, tool-loop chunks."""

import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.specialized.risk_analyst.context import (
    coverage_notes,
    locate_item1a,
    resolve_prior_year,
    section_not_provided,
    tool_chunks,
)
from tests.risk_analyst.item1a_fixtures import CONCENTRATION, item1a_chunk

LOCATE_PATH = "app.agents.specialized.risk_analyst.context.locate_section_chunks"


class TestSectionNotProvided(unittest.TestCase):
    LONG_BODY = "Our business depends on a small number of suppliers and customers in several regions. " * 6

    def test_not_provided_detection(self):
        self.assertTrue(section_not_provided([item1a_chunk("n", "Not applicable.")]))
        self.assertTrue(section_not_provided([item1a_chunk("n", "As a smaller reporting company, we are not required to provide the information required by this Item.")]))
        self.assertTrue(section_not_provided([]))
        self.assertFalse(section_not_provided([item1a_chunk("r", self.LONG_BODY * 10 + " Not applicable to our Japanese subsidiaries.")]))
        self.assertFalse(section_not_provided([item1a_chunk("r", CONCENTRATION)]))


class TestPriorYear(unittest.TestCase):
    def test_latest_filed_year_before_the_target(self):
        self.assertEqual(resolve_prior_year(2025, [2025, 2023, 2022]), 2023)

    def test_no_earlier_filing_means_no_prior_year(self):
        self.assertIsNone(resolve_prior_year(2025, [2025]))

    def test_unknown_filed_years_assume_the_year_before(self):
        self.assertEqual(resolve_prior_year(2025, None), 2024)


class TestLocateItem1a(unittest.TestCase):
    def test_a_failed_lookup_reads_as_not_found(self):
        with patch(LOCATE_PATH, side_effect=RuntimeError("db down")):
            self.assertEqual(locate_item1a("TEST", 2026), ([], "not_found"))

    def test_asks_for_item_1a(self):
        with patch(LOCATE_PATH, return_value=([], "item_label")) as locate:
            locate_item1a("TEST", 2026)
        locate.assert_called_once_with("TEST", 2026, "Item 1A")


class TestCoverageNotes(unittest.TestCase):
    def test_full_section_with_a_prior_year_needs_no_note(self):
        self.assertEqual(coverage_notes("TEST", 2026, "full_section", "item_label", 2025, True), [])

    def test_retrieval_fallback_and_missing_prior_year_are_noted_in_order(self):
        notes = coverage_notes("TEST", 2026, "rag_fallback", "not_found", 2025, False)
        self.assertEqual(len(notes), 2)
        self.assertIn("could not be located in the TEST FY2026 10-K", notes[0])
        self.assertIn("Prior-year (FY2025) Item 1A unavailable", notes[1])

    def test_heading_location_note_replaces_the_budget_note(self):
        notes = coverage_notes("TEST", 2026, "budgeted_section", "heading_boundary", None, False)
        self.assertEqual(len(notes), 1)
        self.assertIn("section headings", notes[0])

    def test_budgeted_section_is_noted(self):
        notes = coverage_notes("TEST", 2026, "budgeted_section", "item_label", None, False)
        self.assertEqual(notes, ["Item 1A exceeded the context budget; non-summary paragraphs were truncated to their lead text."])


class TestToolChunks(unittest.TestCase):
    def test_collects_retrieved_chunks_once_and_skips_other_messages(self):
        payload = [{"chunk_id": "c1", "content": "Item 3 text"}, {"chunk_id": "c1", "content": "duplicate"},
                   {"chunk_id": "c2", "content": 7}]
        messages = [
            AIMessage(content="thinking"),
            ToolMessage(content=json.dumps(payload), tool_call_id="t1"),
            ToolMessage(content='{"chunk_id": "c3", "content": "single"}', tool_call_id="t2"),
            ToolMessage(content='"chunk_id" "content" not json', tool_call_id="t3"),
        ]
        chunks = tool_chunks(messages)
        self.assertEqual(sorted(chunks), ["c1", "c3"])
        self.assertEqual(chunks["c1"]["content"], "Item 3 text")


if __name__ == "__main__":
    unittest.main()
