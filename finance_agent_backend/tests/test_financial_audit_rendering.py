"""Rendering tests for the upgraded financial audit presentation layer.

Validates:
1. The Lead Synthesizer's financial audit context renders the VERIFIED TABLES from a deterministic
   TSLA audit (correct liquid cash, net cash, share count source, severity-ranked findings).
2. The financial_audit_only synthesis mandate and instruction carry the 9-section structure.
3. The ingestion parser emits a compact statement table instead of `Col_n` columns for split-cell HTML.
"""

import unittest

from bs4 import BeautifulSoup

from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.prompts import render_prompt
from app.agents.tools.financial_math_tools import run_audit_from_inputs
from app.agents.tools.statement_extraction import build_statement_grids, extract_audit_inputs
from app.services.ingestion.sec_parser import SECParser
from tests.statement_fixtures import tsla_payloads


def _tsla_audit_output() -> FinancialAuditOutput:
    extraction = extract_audit_inputs(build_statement_grids(tsla_payloads()), 2025, [2023, 2024, 2025])
    data = run_audit_from_inputs(extraction.to_tool_kwargs())
    data.update({
        "ticker": "TSLA",
        "fiscal_year": 2025,
        "auditor_summary": "**Earnings Quality** Operating income $4,355M.",
        "restatement_notes": ["Single filing available; cross-filing restatement review not possible."],
        "data_quality": {
            "extraction_mode": "deterministic",
            "completeness_score": extraction.completeness_score,
            "identity_checks": extraction.identity_checks,
        },
        "income_statement_markdown_table": extraction.clean_tables["income_statement"],
    })
    return FinancialAuditOutput.model_validate(data)


class TestFinancialAuditRendering(unittest.TestCase):

    def test_01_synthesizer_context_tables(self):
        prompt = render_prompt(
            "lead_synthesizer",
            ticker="TSLA",
            company_name="Tesla, Inc.",
            fiscal_year=2025,
            financial_audit=_tsla_audit_output(),
            query_type="financial_audit_only",
            user_query="Audit Tesla's financial statements",
        )
        for marker in [
            "VERIFIED TABLE A", "VERIFIED TABLE B", "VERIFIED TABLE C", "VERIFIED TABLE D",
            "VERIFIED TABLE E", "VERIFIED TABLE F",
            "| Metric | FY2023 | FY2024 | FY2025 |",
            "| Revenue | 96,773.0 | 97,690.0 | 94,827.0 |",
            "| Total liquid cash | $44,059.0M |",
            "| Net debt (negative = net cash) | $-35,683.0M |",
            "| Valuation share count | 3,751.0M (period_end_basic_exceeds_weighted_diluted) |",
            "| HIGH | TAX_ANOMALY | 2023 |",
            "Completeness: 100%",
            "accounting identity checks passed",
            "# Financial Statement Audit & Quality of Earnings: Tesla, Inc. (TSLA)",
            "| Total revenues | 94,827 | 97,690 | 96,773 |",
        ]:
            self.assertIn(marker, prompt)
        # Table rows must be on separate lines (macro newline handling).
        self.assertIn("| Revenue | 96,773.0 | 97,690.0 | 94,827.0 |\n| Revenue growth % |", prompt)
        self.assertNotIn("forensic_red_flags", prompt)

    def test_02_financial_audit_instruction(self):
        instruction = render_prompt(
            "prompt_synthesizer_instruction.j2",
            ticker="TSLA",
            company_name="Tesla, Inc.",
            fiscal_year=2025,
            query_type="financial_audit_only",
        )
        self.assertIn("# Financial Statement Audit & Quality of Earnings: Tesla, Inc. (TSLA)", instruction)
        self.assertIn("VERIFIED TABLES A–F", instruction)

    def test_03_ingestion_parser_collapses_split_cells(self):
        html = """
        <table>
          <tr><td></td><td colspan="2">2025</td><td></td><td colspan="2">2024</td></tr>
          <tr><td>Total revenues</td><td>$</td><td>94,827</td><td></td><td>$</td><td>97,690</td></tr>
          <tr><td>Interest expense</td><td>(338</td><td>)</td><td></td><td>(350</td><td>)</td></tr>
          <tr><td>Net income</td><td>$</td><td>3,855</td><td></td><td>$</td><td>7,153</td></tr>
        </table>
        """
        table = BeautifulSoup(html, "html.parser").find("table")
        markdown, grid, _title = SECParser().table_to_markdown(table)
        self.assertIn("| Line item | 2025 | 2024 |", markdown)
        self.assertIn("| Total revenues | 94,827 | 97,690 |", markdown)
        self.assertIn("| Interest expense | (338) | (350) |", markdown)
        self.assertNotIn("Col_", markdown)
        self.assertEqual(len(grid), 4)

    def test_04_ingestion_parser_keeps_non_statement_tables(self):
        html = """
        <table>
          <tr><td>Name</td><td>Title</td></tr>
          <tr><td>Jane Doe</td><td>Chief Financial Officer</td></tr>
        </table>
        """
        table = BeautifulSoup(html, "html.parser").find("table")
        markdown, _grid, _title = SECParser().table_to_markdown(table)
        self.assertIn("| Name | Title |", markdown)
        self.assertIn("| Jane Doe | Chief Financial Officer |", markdown)


if __name__ == "__main__":
    unittest.main()
