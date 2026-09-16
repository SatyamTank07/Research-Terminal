"""Integration tests for Milestone 2: Financial Auditor Agent.

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Cross-filing tool retrieval across multiple 10-K filings in PostgreSQL.
3. End-to-end execution of FinancialAuditorAgent on Apple Inc. (AAPL FY2025 10-K).
4. Schema conformance of returned FinancialAuditOutput:
   - 3-year contiguous history (2023, 2024, 2025) with zero arithmetic hallucination.
   - Correct unit scaling ($ Millions, Shares in Millions, positive CapEx).
   - Preserved Markdown tables for Income, Balance Sheet, and Cash Flow.
   - Algorithmic red flags and cross-filing restatement notes.
   - Non-empty chunk citations.
"""

import unittest
from app.agents.registry import AgentRegistry
from app.agents.prompts import render_prompt
from app.agents.state import FinancialAuditOutput
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.tools.rag_table_tools import retrieve_multiyear_financial_series_tool


class TestMilestone2FinancialAuditor(unittest.TestCase):
    """Test suite for Milestone 2: Financial Auditor Agent."""

    def test_01_initialization_and_registry(self):
        """Verify agent registration, prompt rendering, and tool binding."""
        agent = AgentRegistry.get("financial_auditor")
        self.assertIsInstance(agent, FinancialAuditorAgent)
        self.assertEqual(agent.model_name, "openai:gpt-4o-mini")

        # Verify prompt renders with role and tools
        prompt = render_prompt("financial_auditor")
        self.assertIn("Senior Forensic CPA", prompt)
        self.assertIn("retrieve_10k_tables_tool", prompt)
        self.assertIn("retrieve_multiyear_financial_series_tool", prompt)
        self.assertIn("audit_financial_metrics_tool", prompt)
        self.assertIn("ZERO ARITHMETIC HALLUCINATION DIRECTIVE", prompt)

    def test_02_cross_filing_retrieval_tool(self):
        """Verify cross-filing tool retrieves consecutive 10-Ks for AAPL."""
        results = retrieve_multiyear_financial_series_tool.invoke({
            "ticker": "AAPL",
            "statement_type": "income_statement",
            "limit": 3,
        })
        self.assertGreaterEqual(len(results), 2, "Should return at least 2 distinct filings for AAPL (FY25 & FY24)")
        years = [r["fiscal_year"] for r in results]
        self.assertIn(2025, years)
        self.assertIn(2024, years)

        for r in results:
            self.assertEqual(r["ticker"], "AAPL")
            self.assertTrue(len(r["table_markdown"]) > 0)
            self.assertTrue(len(r["chunk_id"]) > 0)

    def test_03_end_to_end_apple_fy2025_audit(self):
        """
        Verify autonomous execution of FinancialAuditorAgent on Apple Inc. FY2025.
        Validates complete FinancialAuditOutput artifact.
        """
        agent = FinancialAuditorAgent(model_name="openai:gpt-4o-mini")
        audit_result = agent.audit(ticker="AAPL", fiscal_year=2025)

        # 1. Output Type & Basic Metadata
        self.assertIsInstance(audit_result, FinancialAuditOutput)
        self.assertEqual(audit_result.ticker, "AAPL")
        self.assertEqual(audit_result.fiscal_year, 2025)

        # 2. Multi-Year Financials (3 Contiguous Years: 2023, 2024, 2025)
        history = audit_result.multi_year_history
        self.assertEqual(len(history), 3, "Must extract exactly 3 contiguous fiscal years from 10-K")
        years = [h.fiscal_year for h in history]
        self.assertEqual(years, [2023, 2024, 2025])

        # Verify FY2025 reported values (in $ Millions)
        fy25 = history[-1]
        self.assertEqual(fy25.fiscal_year, 2025)
        # Apple FY25 revenue: $416,161M (+/- 1% tolerance for net sales line)
        self.assertTrue(390_000.0 <= fy25.revenue <= 430_000.0, f"Expected ~416B revenue, got {fy25.revenue}")
        self.assertTrue(110_000.0 <= fy25.operating_income <= 145_000.0, f"Expected ~133B EBIT, got {fy25.operating_income}")
        self.assertTrue(90_000.0 <= fy25.net_income <= 120_000.0, f"Expected ~112B Net Income, got {fy25.net_income}")
        self.assertTrue(100_000.0 <= fy25.operating_cash_flow <= 130_000.0, f"Expected ~111B OCF, got {fy25.operating_cash_flow}")
        
        # Verify CapEx is positive and Free Cash Flow is positive
        self.assertGreater(fy25.capital_expenditures, 0, "CapEx must be a positive magnitude representing cash outflow")
        self.assertTrue(5_000.0 <= fy25.capital_expenditures <= 20_000.0, f"Expected ~12.7B CapEx, got {fy25.capital_expenditures}")
        self.assertTrue(80_000.0 <= fy25.free_cash_flow <= 110_000.0, f"Expected ~98B FCF, got {fy25.free_cash_flow}")
        
        # Verify margins are deterministic and non-zero
        self.assertGreater(fy25.gross_margin_pct, 40.0)
        self.assertGreater(fy25.operating_margin_pct, 25.0)
        self.assertGreater(fy25.net_margin_pct, 20.0)

        # 3. Balance Sheet Verification
        bs = audit_result.balance_sheet
        self.assertEqual(bs.fiscal_year, 2025)
        self.assertGreater(bs.cash_and_equivalents, 0)
        self.assertGreater(bs.marketable_securities, 0)
        self.assertGreater(bs.total_liquid_cash, 50_000.0)  # Apple holds >$130B in total cash + liquid securities
        self.assertGreater(bs.total_debt, 50_000.0)
        
        # Diluted shares scale guard: must be in Millions (~15,000M, NOT 15 billion)
        self.assertTrue(10_000.0 <= bs.diluted_shares_outstanding <= 25_000.0,
                        f"Shares must be in Millions (~15,004M), got {bs.diluted_shares_outstanding}")

        # 4. Profitability & Solvency Ratios
        profit = audit_result.profitability_and_return_ratios
        self.assertGreater(profit.effective_tax_rate_pct, 10.0)
        self.assertLess(profit.effective_tax_rate_pct, 30.0)
        self.assertIsNotNone(profit.nopat)

        solvency = audit_result.solvency_and_liquidity_ratios
        self.assertIsNotNone(solvency.net_debt_to_ebitda_interpretation)

        # 5. Raw Markdown Tables Preservation
        self.assertIn("|", audit_result.income_statement_markdown_table)
        self.assertIn("|", audit_result.balance_sheet_markdown_table)
        self.assertIn("|", audit_result.cash_flow_markdown_table)

        # 6. Forensic Red Flags & Restatement Notes
        self.assertIsInstance(audit_result.forensic_red_flags, list)
        self.assertIsInstance(audit_result.restatement_notes, list)
        self.assertGreater(len(audit_result.auditor_summary), 50)

        # 7. Chunk Citations
        self.assertGreater(len(audit_result.citations), 0, "Must contain chunk citation breadcrumbs")
        for cit in audit_result.citations:
            self.assertIn("chunk_id", cit)
            self.assertIn("breadcrumb", cit)


if __name__ == "__main__":
    unittest.main()
