"""Integration and Unit Tests for Valuation Specialist Agent.

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Standalone execution of ValuationSpecialistAgent with synthetic FinancialAuditOutput:
   - WACC derivation and provenance tagging via calculate_wacc_tool.
   - Deterministic DCF valuation bridge: EV = PV(FCF) + PV(TV), Equity = EV - Net Debt.
   - Net Cash Surplus sign handling (negative Net Debt adds to Equity Value).
   - 5x5 sensitivity matrix generation.
   - Implied EV/EBITDA multiple and +/- 10% valuation stance classification.
3. Standalone execution with an indebted firm (positive Net Debt reduces Equity Value).
4. End-to-end 2-agent valuation chain execution:
   FinancialAuditorAgent (extracts audited balance sheet) -> ValuationSpecialistAgent (computes DCF).
"""

import unittest
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.state import (
    DCFValuationOutput,
    FinancialAuditOutput,
)
from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    audit_financial_metrics,
)
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.valuation_specialist import ValuationSpecialistAgent


def _create_synthetic_aapl_audit() -> FinancialAuditOutput:
    """Helper creating a verified synthetic FinancialAuditOutput matching Apple FY2025 10-K."""
    annual_financials = [
        AnnualFinancialInput(
            fiscal_year=2023,
            revenue=383285.0,
            gross_profit=169148.0,
            operating_income=114301.0,
            pretax_income=113736.0,
            income_tax_expense=16741.0,
            net_income=96995.0,
            operating_cash_flow=110543.0,
            capital_expenditures=10959.0,
        ),
        AnnualFinancialInput(
            fiscal_year=2024,
            revenue=391035.0,
            gross_profit=180683.0,
            operating_income=123216.0,
            pretax_income=123485.0,
            income_tax_expense=29749.0,
            net_income=93736.0,
            operating_cash_flow=118254.0,
            capital_expenditures=9447.0,
        ),
        AnnualFinancialInput(
            fiscal_year=2025,
            revenue=416161.0,
            gross_profit=195201.0,
            operating_income=133050.0,
            pretax_income=132717.0,
            income_tax_expense=20707.0,
            net_income=112010.0,
            operating_cash_flow=111482.0,
            capital_expenditures=12715.0,
            depreciation_amortization=11445.0,
        ),
    ]
    balance_sheet_in = BalanceSheetInput(
        fiscal_year=2025,
        cash_and_equivalents=35934.0,
        marketable_securities=96486.0,
        short_term_debt=10912.0,
        long_term_debt=87745.0,
        stockholders_equity=53736.0,
        diluted_shares_outstanding=15004.7,
        current_assets=154388.0,
        current_liabilities=145308.0,
    )
    math_res = audit_financial_metrics(annual_financials, balance_sheet_in)
    math_res["ticker"] = "AAPL"
    math_res["fiscal_year"] = 2025
    math_res["auditor_summary"] = "Audited financial statements for Apple Inc. FY2025."
    return FinancialAuditOutput.model_validate(math_res)


class TestValuationSpecialistAgent(unittest.TestCase):
    """Test suite for Valuation Specialist Agent."""

    def test_01_initialization_and_registry(self):
        """Verify agent registration, prompt rendering, and tool bindings."""
        agent = AgentRegistry.get("valuation_specialist")
        self.assertIsInstance(agent, ValuationSpecialistAgent)
        self.assertEqual(agent.model_name, "openai:gpt-4o-mini")

        prompt = render_prompt("valuation_specialist")
        self.assertIn("Institutional Valuation Director", prompt)
        self.assertIn("calculate_wacc_tool", prompt)
        self.assertIn("calculate_dcf_tool", prompt)
        self.assertIn("ZERO ARITHMETIC HALLUCINATION DIRECTIVE", prompt)
        self.assertIn("CRITICAL NET DEBT SIGN CONVENTION", prompt)
        self.assertIn("DCFValuationOutput", prompt)

    def test_02_synthetic_standalone_valuation_apple(self):
        """Verify ValuationSpecialistAgent on synthetic Apple FY25 data with Net Cash Surplus."""
        agent = ValuationSpecialistAgent()
        audit_output = _create_synthetic_aapl_audit()

        projected_fcfs = [105000.0, 112000.0, 120000.0, 128000.0, 136000.0]
        beta = 1.10
        share_price = 235.0
        terminal_growth = 0.025

        result = agent.value(
            ticker="AAPL",
            fiscal_year=2025,
            financial_audit=audit_output,
            projected_fcfs=projected_fcfs,
            beta=beta,
            share_price=share_price,
            terminal_growth_rate=terminal_growth,
        )

        self.assertIsInstance(result, DCFValuationOutput)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.fiscal_year, 2025)
        self.assertEqual(result.projected_fcfs, projected_fcfs)
        self.assertEqual(result.terminal_growth_rate, terminal_growth)
        self.assertEqual(result.discounting_convention, "mid_year")

        # 1. WACC audit breakdown verification
        self.assertIsNotNone(result.wacc_audit)
        self.assertGreater(result.wacc_audit.wacc, 0.08)
        self.assertLess(result.wacc_audit.wacc, 0.12)
        self.assertAlmostEqual(result.wacc_audit.cost_of_equity_pct, 9.70, places=1)
        self.assertIn("| WACC Parameter |", result.wacc_audit.formula_breakdown_markdown)

        # 2. Valuation Bridge Verification (Zero Math Hallucination)
        self.assertEqual(result.net_debt, -33763.0)
        self.assertEqual(result.diluted_shares, 15004.7)

        # Equity Value must equal EV - Net Debt
        expected_equity = result.enterprise_value - result.net_debt
        self.assertAlmostEqual(result.equity_value, expected_equity, places=1)

        # Because Net Debt is negative (cash surplus), Equity Value > Enterprise Value
        self.assertGreater(result.equity_value, result.enterprise_value)

        # Fair Value Per Share must equal Equity Value / Diluted Shares
        expected_fair_value = result.equity_value / result.diluted_shares
        self.assertAlmostEqual(result.implied_fair_value_per_share, round(expected_fair_value, 2), places=1)

        # 3. Terminal value percentage sanity check
        self.assertGreater(result.terminal_value_pct_of_ev, 50.0)
        self.assertLess(result.terminal_value_pct_of_ev, 90.0)

        # 4. Sensitivity matrix
        self.assertIn("| WACC \\ Growth |", result.sensitivity_matrix_markdown)
        self.assertIn("*(Base)*", result.sensitivity_matrix_markdown)

        # 5. Share price, upside/downside & stance
        self.assertEqual(result.current_share_price, 235.0)
        self.assertIsNotNone(result.upside_downside_pct)
        self.assertIn(result.valuation_stance, ["Undervalued", "Fairly Valued", "Overvalued"])

        # 6. Implied EV/EBITDA multiple cross-check (True EBITDA = EBIT + D&A)
        self.assertIsNotNone(result.implied_ev_ebitda)
        self.assertEqual(result.ev_ebitda_source, "derived_from_10k_ebit_plus_depreciation")
        expected_ebitda = 133050.0 + 11445.0  # 144,495.0
        self.assertAlmostEqual(
            result.implied_ev_ebitda,
            round(result.enterprise_value / expected_ebitda, 2),
            places=2,
        )
        self.assertGreater(result.implied_ev_ebitda, 5.0)
        self.assertLess(result.implied_ev_ebitda, 40.0)

        # 7. Valuation summary
        self.assertTrue(len(result.valuation_summary) > 50)

    def test_03_indebted_firm_standalone_valuation(self):
        """Verify ValuationSpecialistAgent with an indebted firm (positive Net Debt reduces Equity Value)."""
        agent = ValuationSpecialistAgent()

        annual_financials = [
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=20000.0,
                gross_profit=10000.0,
                operating_income=5000.0,
                net_income=3000.0,
                operating_cash_flow=4000.0,
                capital_expenditures=1000.0,
            )
        ]
        balance_sheet_in = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=2000.0,
            marketable_securities=1000.0,
            short_term_debt=2000.0,
            long_term_debt=18000.0,
            stockholders_equity=10000.0,
            diluted_shares_outstanding=500.0,
        )
        math_res = audit_financial_metrics(annual_financials, balance_sheet_in)
        math_res["ticker"] = "IND"
        math_res["fiscal_year"] = 2025
        math_res["auditor_summary"] = "Audited financials for IND."
        indebted_audit = FinancialAuditOutput.model_validate(math_res)

        projected_fcfs = [3200.0, 3400.0, 3600.0, 3800.0, 4000.0]
        beta = 1.20
        share_price = 45.0

        result = agent.value(
            ticker="IND",
            fiscal_year=2025,
            financial_audit=indebted_audit,
            projected_fcfs=projected_fcfs,
            beta=beta,
            share_price=share_price,
            terminal_growth_rate=0.025,
        )

        self.assertEqual(result.net_debt, 17000.0)
        # Because Net Debt is positive (indebted), Equity Value < Enterprise Value
        self.assertLess(result.equity_value, result.enterprise_value)
        self.assertAlmostEqual(result.equity_value, result.enterprise_value - result.net_debt, places=1)

        # Because D&A was omitted in IND input, implied_ev_ebitda and source must be None (zero silent fallback to EBIT)
        self.assertIsNone(result.implied_ev_ebitda)
        self.assertIsNone(result.ev_ebitda_source)

    def test_04_end_to_end_2agent_chain_aapl(self):
        """Verify end-to-end 2-agent chain: FinancialAuditor -> ValuationSpecialist on AAPL FY25."""
        auditor = FinancialAuditorAgent()
        audit_res = auditor.audit(ticker="AAPL", fiscal_year=2025)

        self.assertIsInstance(audit_res, FinancialAuditOutput)
        self.assertEqual(audit_res.ticker, "AAPL")
        self.assertEqual(audit_res.fiscal_year, 2025)

        # Feed audited output directly into ValuationSpecialist
        val_agent = ValuationSpecialistAgent()
        val_res = val_agent.value(
            ticker="AAPL",
            fiscal_year=2025,
            financial_audit=audit_res,
            projected_fcfs=[105000.0, 112000.0, 120000.0, 128000.0, 136000.0],
            beta=1.10,
            share_price=235.0,
            terminal_growth_rate=0.025,
        )

        self.assertIsInstance(val_res, DCFValuationOutput)
        self.assertEqual(val_res.ticker, "AAPL")
        self.assertEqual(val_res.fiscal_year, 2025)
        self.assertEqual(val_res.net_debt, audit_res.balance_sheet.net_debt)
        self.assertEqual(val_res.diluted_shares, audit_res.balance_sheet.diluted_shares_outstanding)

        # Verify mathematical fidelity
        self.assertAlmostEqual(val_res.equity_value, val_res.enterprise_value - val_res.net_debt, places=1)
        self.assertAlmostEqual(
            val_res.implied_fair_value_per_share,
            round(val_res.equity_value / val_res.diluted_shares, 2),
            places=1,
        )
        self.assertIn("*(Base)*", val_res.sensitivity_matrix_markdown)

        # Verify EV/EBITDA multiple source and value aligns with whether D&A was audited
        if (
            audit_res.solvency_and_liquidity_ratios
            and audit_res.solvency_and_liquidity_ratios.ebitda is not None
        ):
            self.assertIsNotNone(val_res.implied_ev_ebitda)
            self.assertEqual(val_res.ev_ebitda_source, "derived_from_10k_ebit_plus_depreciation")
        else:
            self.assertIsNone(val_res.implied_ev_ebitda)
            self.assertIsNone(val_res.ev_ebitda_source)


if __name__ == "__main__":
    unittest.main()
