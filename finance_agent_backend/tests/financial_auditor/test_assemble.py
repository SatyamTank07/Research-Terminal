"""Output assembly: the authoritative audit overwrites LLM numbers; data quality, tables and citations come from code."""

import json
import unittest

from langchain_core.messages import ToolMessage

from app.agents.specialized.financial_auditor.assemble import (
    assemble_audit,
    clean_tables,
    merge_citations,
    merge_data_quality_findings,
)
from app.agents.specialized.financial_auditor.context import build_statement_grids, extract_audit_inputs
from app.agents.specialized.financial_auditor.modules.statement_integrity import (
    AuditDataQuality,
    assess_statement_integrity,
)
from tests.statement_fixtures import tsla_payloads


def _finding(code, severity, year=2025):
    return {"code": code, "severity": severity, "fiscal_year": year, "metric": "m", "value": None,
            "threshold": None, "message": code}


def _assemble(data, audit=None, extraction=None, tables=None, mode="deterministic"):
    integrity = assess_statement_integrity(extraction) if extraction is not None else None
    return assemble_audit(data, audit=audit, log=["log line"], stale=False, extraction=extraction, integrity=integrity,
                          extraction_mode=mode, prefetched_tables=tables, messages=[])


class TestAssembleAudit(unittest.TestCase):
    def test_audit_numbers_replace_whatever_the_llm_wrote(self):
        data = _assemble({"cost_structure": "llm guess", "auditor_summary": "kept"},
                         audit={"cost_structure": ["code"], "rejected_normalization_adjustments": []})
        self.assertEqual(data["cost_structure"], ["code"])
        self.assertEqual(data["auditor_summary"], "kept")
        self.assertNotIn("rejected_normalization_adjustments", data)

    def test_data_quality_records_the_reconciliation_log(self):
        data = _assemble({}, mode="llm_fallback")
        self.assertEqual(data["data_quality"]["reconciliation_log"], ["log line"])
        self.assertEqual(data["data_quality"]["extraction_mode"], "llm_fallback")

    def test_restatement_notes_are_written_by_code_only_when_the_llm_wrote_none(self):
        extraction = extract_audit_inputs(build_statement_grids(tsla_payloads()), 2025, [2023, 2024, 2025])
        self.assertEqual(_assemble({"restatement_notes": ["llm note"]}, extraction=extraction)["restatement_notes"],
                         ["llm note"])
        self.assertEqual(_assemble({}, extraction=extraction)["restatement_notes"],
                         ["Single filing available; cross-filing restatement review not possible."])

    def test_parsed_statements_become_the_markdown_tables(self):
        extraction = extract_audit_inputs(build_statement_grids(tsla_payloads()), 2025, [2023, 2024, 2025])
        data = _assemble({}, extraction=extraction)
        self.assertIn("| Line item | FY2025 | FY2024 | FY2023 |", data["income_statement_markdown_table"])


class TestMergeDataQualityFindings(unittest.TestCase):
    def test_old_data_quality_findings_are_replaced_and_new_ones_join_after_their_peers(self):
        dq = AuditDataQuality(extraction_mode="llm_fallback")
        merged = merge_data_quality_findings(
            [_finding("TAX_ANOMALY", "high"), _finding("DATA_QUALITY", "medium"), _finding("UNDERINVESTMENT", "info")],
            dq,
        )
        self.assertEqual([(f["code"], f["severity"]) for f in merged],
                         [("TAX_ANOMALY", "high"), ("UNDERINVESTMENT", "info"), ("DATA_QUALITY", "info")])


class TestCleanTables(unittest.TestCase):
    def test_without_a_parse_the_newest_retrieved_table_is_used(self):
        raw = "| not | a statement |\n| --- | --- |\n| x | y |"
        tables = {"income_statement": [{"fiscal_year": 2024, "table_markdown": "old"},
                                       {"fiscal_year": 2025, "table_markdown": raw}]}
        self.assertEqual(clean_tables(None, tables, []), {"income_statement": raw})

    def test_tables_the_llm_retrieved_are_used_too(self):
        msg = ToolMessage(name="retrieve_10k_tables_tool", tool_call_id="t", content=json.dumps(
            [{"statement_type": "cash_flow", "fiscal_year": 2025, "table_markdown": "cf"}]))
        self.assertEqual(clean_tables(None, {}, [msg]), {"cash_flow": "cf"})


class TestMergeCitations(unittest.TestCase):
    def test_fetched_statements_come_first_and_each_chunk_appears_once(self):
        tables = {"balance_sheet": [{"chunk_id": "bs", "item": "Item 8", "fiscal_year": 2025}]}
        merged = merge_citations([{"chunk_id": "llm"}, {"chunk_id": "bs"}, {"chunk_id": None}], tables)
        self.assertEqual([c["chunk_id"] for c in merged], ["bs", "llm"])
        self.assertEqual(merged[0]["statement_type"], "balance_sheet")


if __name__ == "__main__":
    unittest.main()
