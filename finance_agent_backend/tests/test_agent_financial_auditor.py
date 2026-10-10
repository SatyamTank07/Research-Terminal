"""Tests for the deterministic-first Financial Auditor Agent (mocked LLM, zero external API calls).

Validates:
1. Registration, prompt contract and per-run tool binding.
2. Cross-filing retrieval tool against the ingested catalog.
3. Deterministic mode: parsed statements produce the baseline audit even when the LLM never calls a tool.
4. Deterministic mode: the LAST refine tool result is authoritative and every refinement is logged.
5. LLM fallback: inputs the LLM omitted (e.g. short-term investments) are gap-filled from parsed rows.
6. LLM fallback without any parsed statements.
7. Audit window expansion across contiguous filings.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.state import FinancialAuditOutput
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.tools.financial_math_tools import run_audit_from_inputs
from app.agents.tools.rag_table_tools import retrieve_multiyear_financial_series_tool
from app.agents.tools.statement_extraction import build_statement_grids, extract_audit_inputs
from tests.statement_fixtures import tsla_payloads

SUMMARY_JSON = json.dumps({
    "auditor_summary": (
        "**Earnings Quality** Operating income fell to $4,355M.\n\n**Cash Flow Durability** FCF $6,220M.\n\n"
        "**Margins & Cost Structure** Operating margin 4.59%.\n\n**Balance Sheet & Returns** Net cash $35,683M.\n\n"
        "**Forensic Findings & Monitoring** Share count +16.64%."
    ),
    "restatement_notes": [],
    "citations": [],
})


def _mock_agent(messages):
    active = MagicMock()
    active.invoke.return_value = {"messages": messages}
    return active


def _tsla_baseline_inputs():
    return extract_audit_inputs(build_statement_grids(tsla_payloads()), 2025, [2023, 2024, 2025]).to_tool_kwargs()


class TestFinancialAuditorAgent(unittest.TestCase):

    def test_01_initialization_prompt_and_tool_binding(self):
        agent = AgentRegistry.get("financial_auditor")
        self.assertIsInstance(agent, FinancialAuditorAgent)
        self.assertEqual(agent.model_name, "openai:gpt-4o-mini")

        prompt = render_prompt("financial_auditor")
        for marker in ["Senior Forensic CPA", "refine_financial_audit_tool", "audit_financial_metrics_tool",
                       "<auditor_summary_contract>", "normalization_adjustments", "**Earnings Quality**"]:
            self.assertIn(marker, prompt)

        agent._configure_tools("deterministic", _tsla_baseline_inputs())
        self.assertIn("refine_financial_audit_tool", [t.name for t in agent.get_tools()])
        self.assertNotIn("audit_financial_metrics_tool", [t.name for t in agent.get_tools()])
        agent._configure_tools("llm_fallback", None)
        self.assertIn("audit_financial_metrics_tool", [t.name for t in agent.get_tools()])

    def test_02_cross_filing_retrieval_tool(self):
        results = retrieve_multiyear_financial_series_tool.invoke({
            "ticker": "AAPL",
            "statement_type": "income_statement",
            "limit": 3,
        })
        self.assertGreaterEqual(len(results), 2, "Should return at least 2 distinct filings for AAPL (FY25 & FY24)")
        years = [r["fiscal_year"] for r in results]
        self.assertIn(2025, years)
        self.assertIn(2024, years)

    @patch.object(FinancialAuditorAgent, "_prefetch_statement_tables", return_value=tsla_payloads())
    @patch.object(FinancialAuditorAgent, "_get_or_create_agent")
    def test_03_deterministic_baseline_without_tool_calls(self, mock_get_agent, _prefetch):
        mock_get_agent.return_value = _mock_agent([AIMessage(content=SUMMARY_JSON)])

        result = FinancialAuditorAgent().audit(ticker="tsla", fiscal_year=2025)

        self.assertIsInstance(result, FinancialAuditOutput)
        self.assertEqual(result.ticker, "TSLA")
        self.assertEqual([h.fiscal_year for h in result.multi_year_history], [2023, 2024, 2025])

        bs = result.balance_sheet
        self.assertEqual(bs.marketable_securities, 27546.0)
        self.assertEqual(bs.net_debt, -35683.0)
        self.assertEqual(bs.valuation_shares_outstanding, 3751.0)
        self.assertAlmostEqual(result.profitability_and_return_ratios.effective_tax_rate_pct, 26.96, places=2)
        self.assertEqual(result.multi_year_history[-1].net_income, 3794.0)
        self.assertAlmostEqual(result.multi_year_history[-1].gross_margin_pct, 18.03, places=2)

        codes = {f.code for f in result.forensic_findings}
        self.assertTrue({"SHARE_DILUTION", "CAPEX_DRIVEN_FCF", "TAX_ANOMALY", "SBC_INTENSITY"} <= codes)
        self.assertNotIn("DATA_QUALITY", codes)

        dq = result.data_quality
        self.assertEqual(dq.extraction_mode, "deterministic")
        self.assertEqual(dq.completeness_score, 1.0)
        self.assertFalse(dq.summary_may_be_stale)
        self.assertTrue(any("baseline retained" in line for line in dq.reconciliation_log))
        self.assertIn("FY2025.revenue", dq.field_provenance)

        self.assertIn("| Total revenues | 94,827 | 97,690 | 96,773 |", result.income_statement_markdown_table)
        self.assertNotIn("Col_", result.balance_sheet_markdown_table)
        self.assertEqual(result.restatement_notes,
                         ["Single filing available; cross-filing restatement review not possible."])
        self.assertEqual({c["chunk_id"] for c in result.citations}, {"tsla-is", "tsla-bs", "tsla-cf"})
        self.assertTrue(result.auditor_summary.startswith("**Earnings Quality**"))

    @patch.object(FinancialAuditorAgent, "_prefetch_statement_tables", return_value=tsla_payloads())
    @patch.object(FinancialAuditorAgent, "_get_or_create_agent")
    def test_04_last_refinement_wins_and_is_logged(self, mock_get_agent, _prefetch):
        refine = FinancialAuditorAgent._build_refine_tool(_tsla_baseline_inputs())
        first = refine.invoke({"normalization_adjustments": [
            {"fiscal_year": 2025, "label": "Placeholder", "amount": 100.0, "direction": "inflated_reported_earnings"},
        ]})
        final = refine.invoke({
            "normalization_adjustments": [
                {"fiscal_year": 2025, "label": "Automotive regulatory credits", "amount": 1993.0,
                 "direction": "inflated_reported_earnings", "affects": "operating_income",
                 "rationale": "Non-core credit sales", "source_chunk_id": "tsla-is"},
            ],
            "corrections": [
                {"target": "balance_sheet", "fiscal_year": 2025, "field": "current_liabilities",
                 "value": 31714.0, "reason": "Confirmed against balance sheet", "source_chunk_id": "tsla-bs"},
            ],
        })
        mock_get_agent.return_value = _mock_agent([
            ToolMessage(name="refine_financial_audit_tool", content=json.dumps(first), tool_call_id="r1"),
            ToolMessage(name="refine_financial_audit_tool", content=json.dumps(final), tool_call_id="r2"),
            AIMessage(content=SUMMARY_JSON),
        ])

        result = FinancialAuditorAgent().audit(ticker="TSLA", fiscal_year=2025)

        fy25 = result.earnings_quality[-1]
        self.assertEqual(fy25.normalized_operating_income, 2362.0)
        self.assertEqual([a.label for a in fy25.adjustments], ["Automotive regulatory credits"])
        log = " ".join(result.data_quality.reconciliation_log)
        self.assertIn("Normalized FY2025 operating_income", log)
        self.assertIn("Corrected balance_sheet FY2025.current_liabilities", log)
        materiality = [f for f in result.forensic_findings
                       if f.metric == "normalization_adjustments_pct_of_operating_income"]
        self.assertEqual(len(materiality), 1)

    @patch.object(FinancialAuditorAgent, "_get_or_create_agent")
    def test_05_llm_fallback_gap_fills_omitted_inputs(self, mock_get_agent):
        partial = tsla_payloads()
        partial["cash_flow"] = []  # cash flow statement unavailable -> deterministic mode impossible

        llm_args = {
            "annual_financials": [
                {"fiscal_year": 2024, "revenue": 97690.0, "operating_income": 7076.0, "net_income": 7091.0,
                 "operating_cash_flow": 14923.0, "capital_expenditures": 11342.0},
                {"fiscal_year": 2025, "revenue": 94827.0, "operating_income": 4355.0, "net_income": 3794.0,
                 "operating_cash_flow": 14747.0, "capital_expenditures": 8527.0},
            ],
            "balance_sheet": {"fiscal_year": 2025, "cash_and_equivalents": 16513.0, "marketable_securities": 0.0,
                              "short_term_debt": 1640.0, "long_term_debt": 6736.0,
                              "stockholders_equity": 82137.0, "weighted_diluted_shares": 3528.0},
        }
        llm_result = run_audit_from_inputs(llm_args)
        self.assertEqual(llm_result["balance_sheet"]["net_debt"], -8137.0)  # the original defect

        mock_get_agent.return_value = _mock_agent([
            AIMessage(content="", tool_calls=[{"name": "audit_financial_metrics_tool", "args": llm_args, "id": "a1"}]),
            ToolMessage(name="audit_financial_metrics_tool", content=json.dumps(llm_result), tool_call_id="a1"),
            AIMessage(content=SUMMARY_JSON),
        ])

        with patch.object(FinancialAuditorAgent, "_prefetch_statement_tables", return_value=partial):
            result = FinancialAuditorAgent().audit(ticker="TSLA", fiscal_year=2025)

        dq = result.data_quality
        self.assertEqual(dq.extraction_mode, "llm_fallback")
        self.assertTrue(dq.summary_may_be_stale)
        self.assertEqual(result.balance_sheet.marketable_securities, 27546.0)
        self.assertEqual(result.balance_sheet.net_debt, -35683.0)
        self.assertEqual(result.balance_sheet.valuation_shares_outstanding, 3751.0)
        self.assertEqual(result.multi_year_history[-1].gross_profit, 17094.0)
        self.assertAlmostEqual(result.profitability_and_return_ratios.effective_tax_rate_pct, 26.96, places=2)
        self.assertEqual(result.profitability_and_return_ratios.capital_basis, "average")
        log = " ".join(dq.reconciliation_log)
        self.assertIn("Gap-filled FY2025.balance_sheet.marketable_securities", log)
        self.assertIn("Added FY2024 prior balance sheet", log)
        self.assertIn("FY2025.operating_cash_flow", dq.missing_fields)
        self.assertIn("DATA_QUALITY", {f.code for f in result.forensic_findings})

    @patch.object(FinancialAuditorAgent, "_prefetch_statement_tables", return_value={})
    @patch.object(FinancialAuditorAgent, "_get_or_create_agent")
    def test_06_llm_fallback_without_parsed_statements(self, mock_get_agent, _prefetch):
        llm_args = {
            "annual_financials": [
                {"fiscal_year": 2025, "revenue": 1000.0, "operating_income": 200.0, "net_income": 150.0,
                 "operating_cash_flow": 180.0, "capital_expenditures": 40.0},
            ],
            "balance_sheet": {"fiscal_year": 2025, "cash_and_equivalents": 100.0, "stockholders_equity": 500.0,
                              "period_end_shares_outstanding": 50.0},
        }
        mock_get_agent.return_value = _mock_agent([
            AIMessage(content="", tool_calls=[{"name": "audit_financial_metrics_tool", "args": llm_args, "id": "a1"}]),
            ToolMessage(name="audit_financial_metrics_tool", content=json.dumps(run_audit_from_inputs(llm_args)),
                        tool_call_id="a1"),
            AIMessage(content=SUMMARY_JSON),
        ])

        result = FinancialAuditorAgent().audit(ticker="XYZ", fiscal_year=2025)

        self.assertEqual(result.data_quality.extraction_mode, "llm_fallback")
        self.assertIsNone(result.data_quality.completeness_score)
        self.assertFalse(result.data_quality.summary_may_be_stale)
        self.assertEqual(result.balance_sheet.share_count_source, "period_end_basic_only")
        self.assertEqual(result.balance_sheet.net_debt, -100.0)

    @patch.object(FinancialAuditorAgent, "_prefetch_statement_tables", return_value={})
    @patch.object(FinancialAuditorAgent, "execute_structured")
    def test_07_multiyear_available_years_expansion(self, mock_exec, _prefetch):
        mock_exec.return_value = MagicMock(spec=FinancialAuditOutput)
        FinancialAuditorAgent().audit(
            ticker="AMZN",
            fiscal_year=2025,
            available_fiscal_years=[2025, 2024, 2023, 2022, 2021],
        )
        called_query = mock_exec.call_args[0][0]
        self.assertIn("Audit window: 2021, 2022, 2023, 2024, 2025", called_query)
        self.assertIn("EXTRACTION MODE: llm_fallback", called_query)
        self.assertEqual(mock_exec.call_args.kwargs["extraction_mode"], "llm_fallback")

        self.assertEqual(FinancialAuditorAgent._resolve_audit_window(2025, [2025, 2024, 2022]), [2023, 2024, 2025])


if __name__ == "__main__":
    unittest.main()
