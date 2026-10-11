"""SEC filing parser tests: section detection across 10-K heading layouts and filing metadata.

Validates:
1. Headings rendered as one-row tables ("| Item 1A. | Risk Factors |") open their sections.
2. Plain text headings ("Item 1A.    Risk Factors") keep working unchanged.
3. Cross-references ("... Item 8 of Part II ...", "Item 15 of this report") never open a section.
4. Table-of-contents entries (page-numbered, or listed before the real headings, or a TOC with no
   body headings at all) never open a section; real empty sections ("Item 6. [Reserved]") still do.
5. Headings split across elements, combined "Items 1 and 2" headings, PART + Item in one element.
6. Text sitting next to a table inside a wrapper element is kept.
7. A 10-K missing core items (1, 1A, 7, 8) is flagged.
8. Metadata comes from the filing (dei facts, cover page, EDGAR file name) and fails loudly otherwise.
"""

import os
import shutil
import tempfile
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock, patch

from app.services.ingestion.sec_parser import (
    FilingMetadataError,
    ParsedSection,
    SECParser,
    classify_heading,
    is_heading_title,
    missing_core_sections,
)

DEI_HEADER = """
<ix:header><ix:hidden>
<ix:nonNumeric name="dei:DocumentType" contextRef="c-1">10-K</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c-1">2031</ix:nonNumeric>
<ix:nonNumeric name="dei:TradingSymbol" contextRef="c-1">EXCO</ix:nonNumeric>
<ix:nonNumeric name="dei:EntityRegistrantName" contextRef="c-1">Example Holdings, Inc.</ix:nonNumeric>
<ix:nonNumeric name="dei:EntityCentralIndexKey" contextRef="c-1">0000123456</ix:nonNumeric>
</ix:hidden></ix:header>
"""

PARAGRAPH = "Example Holdings designs, manufactures and sells widgets to customers around the world. " * 4


def _heading_table(number: str, title: str) -> str:
    return (
        '<table><tr><td style="width:10%"></td><td></td></tr>'
        f'<tr><td><span style="font-weight:700">Item&#160;{number}.</span></td>'
        f'<td><span style="font-weight:700">{title}</span></td></tr></table>'
    )


def _div(text: str) -> str:
    return f"<div><span>{text}</span></div>"


def _filing(body: str, header: str = DEI_HEADER) -> str:
    return f"<html><head><title>10-K</title></head><body>{header}{body}</body></html>"


TOC_TABLE = (
    "<table><tr><td></td><td></td><td>Page</td></tr>"
    "<tr><td>PART I</td><td></td><td></td></tr>"
    "<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>"
    "<tr><td>Item 1A.</td><td>Risk Factors</td><td>6</td></tr>"
    "<tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>20</td></tr>"
    "<tr><td>Item 8.</td><td>Financial Statements and Supplementary Data</td><td>35</td></tr>"
    "</table>"
)

STATEMENT_TABLE = (
    "<table><tr><td></td><td>2030</td><td>2031</td></tr>"
    "<tr><td>Net sales</td><td>100</td><td>120</td></tr>"
    "<tr><td>Net income</td><td>10</td><td>12</td></tr></table>"
)


class _ParserCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def parse(self, html: str, filename: str = "exco-20311231.htm") -> Tuple[object, List[ParsedSection]]:
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return SECParser().parse(path)

    @staticmethod
    def items(sections: List[ParsedSection]) -> List[str]:
        return [s.item for s in sections]

    @staticmethod
    def section(sections: List[ParsedSection], item: str) -> ParsedSection:
        return next(s for s in sections if s.item == item)

    @staticmethod
    def text_of(section: ParsedSection) -> str:
        return " ".join(b.content for b in section.blocks)


class TestHeadingClassifier(unittest.TestCase):

    def test_01_item_heading_variants(self):
        self.assertEqual(classify_heading("Item 1A. Risk Factors"), ("item", "Item 1A", "Risk Factors"))
        self.assertEqual(classify_heading("ITEM 1A. RISK FACTORS"), ("item", "Item 1A", "RISK FACTORS"))
        self.assertEqual(classify_heading("Item 7: Management’s Discussion"), ("item", "Item 7", "Management’s Discussion"))
        self.assertEqual(classify_heading("Item\xa01A.\xa0\xa0Risk Factors"), ("item", "Item 1A", "Risk Factors"))
        self.assertEqual(classify_heading("Item 1 A . Risk Factors"), ("item", "Item 1A", "Risk Factors"))
        self.assertEqual(classify_heading("Item 7 — Management’s Discussion"), ("item", "Item 7", "Management’s Discussion"))
        self.assertEqual(classify_heading("Item 1A."), ("item", "Item 1A", ""))
        self.assertEqual(classify_heading("Item 6. [Reserved]"), ("item", "Item 6", "[Reserved]"))

    def test_02_title_letter_is_not_an_item_suffix(self):
        self.assertEqual(classify_heading("Item 1 Business"), ("item", "Item 1", "Business"))
        self.assertEqual(classify_heading("ITEM 1 BUSINESS"), ("item", "Item 1", "BUSINESS"))

    def test_03_combined_items(self):
        self.assertEqual(
            classify_heading("Items 1 and 2. Business and Properties"),
            ("item", "Item 1", "Business and Properties"),
        )

    def test_04_cross_references_rejected(self):
        for text in [
            "Item 8 of Part II, “Financial Statements and Supplementary Data — Note 7.”",
            "Item 15 of this Annual Report on Form 10-K for further discussion",
            "1. Financial statements (see Index in Part II, Item 8 of this report)",
            "See Item 1A of Part I — “Risk Factors.”",
            "Part II, Item 7 of this Form 10-K",
        ]:
            self.assertIsNone(classify_heading(text), text)

    def test_05_non_items_and_long_text_rejected(self):
        self.assertIsNone(classify_heading("Item 5.02 Departure of Directors"))
        self.assertIsNone(classify_heading("Item 99. Exhibits"))
        self.assertIsNone(classify_heading("Item 7. " + "Management discusses results " * 10))

    def test_06_toc_entries_rejected(self):
        self.assertIsNone(classify_heading("Item 1A. Risk Factors 6"))
        self.assertIsNone(classify_heading("Item 8. Financial Statements F-1"))

    def test_07_part_headings(self):
        self.assertEqual(classify_heading("PART I"), ("part", "PART I", ""))
        self.assertEqual(classify_heading("PART II — OTHER INFORMATION"), ("part", "PART II", "OTHER INFORMATION"))
        self.assertEqual(classify_heading("Part IV"), ("part", "PART IV", ""))
        self.assertIsNone(classify_heading("Part of our strategy is to grow."))

    def test_08_heading_title_check(self):
        self.assertTrue(is_heading_title("Risk Factors"))
        self.assertTrue(is_heading_title("“Risk Factors”"))
        self.assertTrue(is_heading_title(""))
        self.assertFalse(is_heading_title("of this report)"))
        self.assertFalse(is_heading_title(", Item 7"))


class TestSectionDetection(_ParserCase):

    def test_01_table_rendered_headings(self):
        """Every heading is a one-row table (the layout that left a filer with only Cover Page / Item 8)."""
        html = _filing(
            _div("ANNUAL REPORT") + TOC_TABLE
            + _div("PART I") + _heading_table("1", "Business") + _div("BUSINESS TEXT " + PARAGRAPH)
            + _heading_table("1A", "Risk Factors") + _div("RISK TEXT " + PARAGRAPH)
            + _heading_table("3", "Legal Proceedings")
            + _div("See Item 8 of Part II, “Financial Statements and Supplementary Data — Note 7.”")
            + _div("LEGAL TEXT " + PARAGRAPH)
            + _div("PART II") + _heading_table("7", "Management’s Discussion and Analysis")
            + _div("MDNA TEXT " + PARAGRAPH)
            + _heading_table("8", "Financial Statements and Supplementary Data") + STATEMENT_TABLE
        )
        _meta, sections = self.parse(html)

        self.assertEqual(self.items(sections), ["Cover Page", "Item 1", "Item 1A", "Item 3", "Item 7", "Item 8"])
        self.assertEqual(self.section(sections, "Item 1A").title, "Risk Factors")
        self.assertEqual(self.section(sections, "Item 1A").part, "PART I")
        self.assertEqual(self.section(sections, "Item 7").part, "PART II")
        self.assertIn("RISK TEXT", self.text_of(self.section(sections, "Item 1A")))
        self.assertIn("MDNA TEXT", self.text_of(self.section(sections, "Item 7")))
        # The cross-reference stays prose inside Item 3 instead of opening "Item 8".
        legal = self.text_of(self.section(sections, "Item 3"))
        self.assertIn("See Item 8 of Part II", legal)
        self.assertIn("LEGAL TEXT", legal)
        # Heading tables are consumed; the TOC table stays a cover-page table; the statement is a table block.
        cover = self.section(sections, "Cover Page")
        self.assertEqual([b.block_type for b in cover.blocks], ["text", "table"])
        self.assertEqual([b.block_type for b in self.section(sections, "Item 8").blocks], ["table"])
        self.assertEqual(missing_core_sections("10-K", sections), [])

    def test_02_regression_text_headings(self):
        """Headings as styled text divs keep the labels and titles the parser has always produced."""
        html = _filing(
            TOC_TABLE + _div("PART I")
            + _div("Item 1.&#160;&#160;&#160;&#160;Business") + _div(PARAGRAPH)
            + _div("Item 1A.&#160;&#160;&#160;&#160;Risk Factors") + _div(PARAGRAPH)
            + _div("PART II")
            + _div("Item 7.&#160;&#160;&#160;&#160;Management’s Discussion and Analysis of Financial Condition")
            + _div("Refer to Item 15 of this Annual Report on Form 10-K for further discussion.")
            + _div(PARAGRAPH)
            + _div("Item 8.&#160;&#160;&#160;&#160;Financial Statements and Supplementary Data") + STATEMENT_TABLE
        )
        _meta, sections = self.parse(html)

        self.assertEqual(self.items(sections), ["Cover Page", "Item 1", "Item 1A", "Item 7", "Item 8"])
        self.assertEqual(
            [(s.part, s.title) for s in sections[1:]],
            [
                ("PART I", "Business"),
                ("PART I", "Risk Factors"),
                ("PART II", "Management’s Discussion and Analysis of Financial Condition"),
                ("PART II", "Financial Statements and Supplementary Data"),
            ],
        )
        self.assertIn("Refer to Item 15", self.text_of(self.section(sections, "Item 7")))

    def test_03_text_toc_before_real_headings(self):
        """A text TOC (no page numbers) is listed again by the real headings, so it is demoted."""
        toc = "".join(
            _div(entry) for entry in
            ["Item 1. Business", "Item 1A. Risk Factors", "Item 7. Management’s Discussion", "Item 8. Financial Statements"]
        )
        html = _filing(
            _div("COVER TEXT " + PARAGRAPH) + toc
            + _div("Item 1. Business") + _div("BUSINESS TEXT " + PARAGRAPH)
            + _div("Item 1A. Risk Factors") + _div("RISK TEXT " + PARAGRAPH)
            + _div("Item 7. Management’s Discussion") + _div("MDNA TEXT " + PARAGRAPH)
            + _div("Item 8. Financial Statements") + STATEMENT_TABLE
        )
        _meta, sections = self.parse(html)

        self.assertEqual(self.items(sections), ["Cover Page", "Item 1", "Item 1A", "Item 7", "Item 8"])
        self.assertIn("Item 8. Financial Statements", self.text_of(sections[0]))
        self.assertIn("BUSINESS TEXT", self.text_of(self.section(sections, "Item 1")))

    def test_04_toc_only_matches(self):
        """Headings that appear only in a TOC spanning the Parts never label the body."""
        toc = "".join(
            _div(entry) for entry in
            ["PART I", "Item 1. Business", "Item 1A. Risk Factors", "PART II", "Item 7. Management’s Discussion",
             "PART III", "Item 10. Directors", "PART IV", "Item 15. Exhibits"]
        )
        html = _filing(_div("COVER TEXT " + PARAGRAPH) + toc + _div("BODY TEXT " + PARAGRAPH))
        with self.assertLogs("finance_agent.ingestion.sec_parser", level="WARNING") as logs:
            _meta, sections = self.parse(html)

        self.assertEqual({s.item for s in sections}, {"Cover Page"})
        self.assertIn("Item 1, Item 1A, Item 7, Item 8", "\n".join(logs.output))

    def test_05_empty_real_sections_are_not_toc(self):
        """Consecutive empty headings within one Part ("[Reserved]", Part III by reference) stay sections."""
        html = _filing(
            _div("Item 5. Market for Common Equity") + _div(PARAGRAPH)
            + _div("Item 6. [Reserved]")
            + _div("Item 7. Management’s Discussion") + _div("MDNA TEXT " + PARAGRAPH)
            + _div("PART III")
            + _div("Item 10. Directors") + _div("Item 11. Executive Compensation")
            + _div("Item 12. Security Ownership") + _div("PART III TEXT " + PARAGRAPH)
        )
        _meta, sections = self.parse(html)

        self.assertIn("MDNA TEXT", self.text_of(self.section(sections, "Item 7")))
        self.assertIn("PART III TEXT", self.text_of(self.section(sections, "Item 12")))
        self.assertEqual(self.section(sections, "Item 12").part, "PART III")

    def test_06_split_and_combined_headings(self):
        html = _filing(
            _div("PART I") + _div("Items 1 and 2.") + _div("Business and Properties") + _div(PARAGRAPH)
            + "<div><div><span>Item 1A.</span></div><div><span>Risk Factors</span></div></div>" + _div(PARAGRAPH)
            + _div("PART II ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS") + _div(PARAGRAPH)
        )
        _meta, sections = self.parse(html)

        self.assertEqual(
            [(s.part, s.item, s.title) for s in sections],
            [
                ("PART I", "Item 1", "Business and Properties"),
                ("PART I", "Item 1A", "Risk Factors"),
                ("PART II", "Item 7", "MANAGEMENT’S DISCUSSION AND ANALYSIS"),
            ],
        )

    def test_07_text_next_to_table_in_wrapper_is_kept(self):
        """Narrative wrapped together with a table (e.g. an iXBRL note text block) is not dropped."""
        html = _filing(
            _div("Item 8. Financial Statements")
            + f"<div><ix:nonNumeric name=\"us-gaap:RevenueTextBlock\"><div><span>NOTE TEXT {PARAGRAPH}</span></div>"
            + STATEMENT_TABLE + "<div><span>AFTER TABLE TEXT</span></div></ix:nonNumeric></div>"
        )
        _meta, sections = self.parse(html)

        item8 = self.section(sections, "Item 8")
        self.assertEqual([b.block_type for b in item8.blocks], ["text", "table", "text"])
        self.assertIn("NOTE TEXT", item8.blocks[0].content)
        self.assertEqual(item8.blocks[2].content, "AFTER TABLE TEXT")

    def test_08_missing_risk_factors_is_flagged(self):
        """A smaller reporting company that omits Item 1A entirely is flagged, not silently accepted."""
        html = _filing(
            _div("Item 1. Business") + _div(PARAGRAPH)
            + _div("Item 7. Management’s Discussion") + _div(PARAGRAPH)
            + _div("Item 8. Financial Statements") + STATEMENT_TABLE
        )
        with self.assertLogs("finance_agent.ingestion.sec_parser", level="WARNING") as logs:
            _meta, sections = self.parse(html)
        self.assertEqual(missing_core_sections("10-K", sections), ["Item 1A"])
        self.assertTrue(any("Item 1A" in line for line in logs.output))

    def test_09_risk_factors_not_applicable_is_present(self):
        html = _filing(
            _div("Item 1. Business") + _div(PARAGRAPH)
            + _div("Item 1A. Risk Factors") + _div("Not applicable for smaller reporting companies.")
            + _div("Item 7. Management’s Discussion") + _div(PARAGRAPH)
            + _div("Item 8. Financial Statements") + STATEMENT_TABLE
        )
        _meta, sections = self.parse(html)
        self.assertEqual(missing_core_sections("10-K", sections), [])
        self.assertEqual(missing_core_sections("10-K/A", []), [])


class TestFilingMetadata(_ParserCase):

    def test_01_dei_facts(self):
        header = DEI_HEADER.replace(
            "</ix:hidden>",
            '<ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="c-1">'
            '<ix:nonNumeric name="dei:CurrentFiscalYearEndDate" contextRef="c-1">September&#160;27</ix:nonNumeric>, 2031'
            "</ix:nonNumeric></ix:hidden>",
        )
        meta = SECParser().extract_metadata(_filing("", header), "/tmp/whatever.htm")
        self.assertEqual(
            (meta.ticker, meta.company_name, meta.filing_type, meta.fiscal_year, meta.period_end_date, meta.cik),
            ("EXCO", "Example Holdings, Inc.", "10-K", 2031, "2031-09-27", "0000123456"),
        )

    def test_02_cover_page_and_file_name_fallbacks(self):
        cover = (
            _div("UNITED STATES SECURITIES AND EXCHANGE COMMISSION") + _div("FORM 10-K")
            + _div("For the fiscal year ended March 31, 2031") + _div("Widget Works Corp.")
            + _div("(Exact name of registrant as specified in its charter)")
        )
        with self.assertLogs("finance_agent.ingestion.sec_parser", level="WARNING"):
            meta = SECParser().extract_metadata(_filing(cover, header=""), "/data/wdgt-20310331.htm")
        self.assertEqual(
            (meta.ticker, meta.company_name, meta.filing_type, meta.fiscal_year, meta.period_end_date),
            ("WDGT", "Widget Works Corp.", "10-K", 2031, "2031-03-31"),
        )

    def test_03_unidentifiable_filing_fails_loudly(self):
        with self.assertRaises(FilingMetadataError):
            SECParser().extract_metadata(_filing(_div("FORM 10-K"), header=""), "/tmp/annual_report.htm")
        no_year = DEI_HEADER.replace("2031", "")
        with self.assertRaises(FilingMetadataError):
            SECParser().extract_metadata(_filing(_div("FORM 10-K"), no_year), "/tmp/report.htm")

    def test_04_no_hardcoded_company_names(self):
        header = DEI_HEADER.replace(
            '<ix:nonNumeric name="dei:EntityRegistrantName" contextRef="c-1">Example Holdings, Inc.</ix:nonNumeric>', ""
        ).replace("EXCO", "AAPL")
        with self.assertRaises(FilingMetadataError):
            SECParser().extract_metadata(_filing(_div("FORM 10-K"), header), "/tmp/aapl-20310930.htm")


class TestPipelineQualityFlag(_ParserCase):

    def test_01_missing_core_sections_reported(self):
        from app.services.ingestion.pipeline import SECIngestionPipeline

        html = _filing(_div("Item 1. Business") + _div(PARAGRAPH))
        path = os.path.join(self.tmpdir, "exco-20311231.htm")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        events = []
        with patch("app.services.ingestion.pipeline.SECBatchEmbedder") as embedder_cls:
            embedder_cls.return_value.embed_texts.side_effect = lambda texts, **_: [[0.0] * 1536 for _ in texts]
            result = SECIngestionPipeline(db).run(path, progress_callback=events.append)

        self.assertEqual(result["missing_core_sections"], ["Item 1A", "Item 7", "Item 8"])
        self.assertIn("no section headings found for Item 1A, Item 7, Item 8", events[-1]["message"])
        db.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
