"""Integration and Unit Tests for Forecasting Analyst Agent (Milestone 4).

Validates:
1. Agent registration in AgentRegistry and prompt template rendering.
2. Standalone execution of ForecastingAnalystAgent with synthetic FinancialAuditOutput:
   - Automatic accounting mode precedence ('comprehensive_line_item' when D&A present).
   - Test-only force_mode override ('simplified_nopat_less_capex').
   - Derivation of 5-year compounding schedule and direct DCF handoff (projected_fcfs).
3. Deterministic CAGR decay fallback mechanism:
   - 75 bps / year annual decay towards 2.75% long-term GDP terminal floor.
   - Tagging guidance_source="historical_cagr_decay".
4. End-to-end 3-agent valuation chain execution:
   FinancialAuditor -> ForecastingAnalyst (produces 5-Yr UFCFs) -> ValuationSpecialist (computes DCF).
"""

import unittest
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.state import (
    FinancialAuditOutput,
    ForecastOutput,
    DCFValuationOutput,
)
from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    audit_financial_metrics,
)
from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
from app.agents.specialized.valuation_specialist import ValuationSpecialistAgent


def _create_synthetic_aapl_audit(include_depreciation: bool = True) -> FinancialAuditOutput:
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
            depreciation_amortization=11445.0 if include_depreciation else None,
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


class TestForecastingAnalystAgent(unittest.TestCase):
    """Test suite for Forecasting Analyst Agent."""

    def test_01_initialization_and_registry(self):
        """Verify agent registration, prompt rendering, and tool bindings."""
        agent = AgentRegistry.get("forecasting_analyst")
        self.assertIsInstance(agent, ForecastingAnalystAgent)
        self.assertEqual(agent.model_name, "openai:gpt-4o-mini")

        prompt = render_prompt("forecasting_analyst")
        self.assertIn("Senior Buy-Side Forecasting Analyst", prompt)
        self.assertIn("calculate_forecast_schedule_tool", prompt)
        self.assertIn("retrieve_10k_narrative_tool", prompt)
        self.assertIn("Zero Arithmetic Hallucination", prompt)

    def test_02_apple_fy2025_standalone_forecast_comprehensive(self):
        """
        Verify standalone forecast execution for Apple FY2025:
        - D&A is present -> automatically resolves to 'comprehensive_line_item'
        - Generates 5-year explicit schedule
        - Produces 5 valid UFCF floats
        """
        agent = ForecastingAnalystAgent()
        audit = _create_synthetic_aapl_audit(include_depreciation=True)

        forecast_out = agent.forecast(
            ticker="AAPL",
            fiscal_year=2025,
            financial_audit=audit,
            horizon_years=5,
        )

        self.assertIsInstance(forecast_out, ForecastOutput)
        self.assertEqual(forecast_out.ticker, "AAPL")
        self.assertEqual(forecast_out.fiscal_year, 2025)
        self.assertEqual(forecast_out.base_revenue, 416161.0)
        self.assertEqual(forecast_out.forecast_horizon_years, 5)
        self.assertEqual(forecast_out.provenance_mode, "comprehensive_line_item")
        self.assertIn(forecast_out.guidance_source, ["md&a_explicit", "historical_cagr_decay"])

        # 5 explicit years
        self.assertEqual(len(forecast_out.projected_fcfs), 5)
        self.assertEqual(len(forecast_out.forecast_schedule), 5)
        for fcf in forecast_out.projected_fcfs:
            self.assertGreater(fcf, 50000.0, "Apple explicit UFCF should exceed $50B/year")

        # Check schedule line items
        y1 = forecast_out.forecast_schedule[0]
        self.assertEqual(y1.projected_year, 2026)
        self.assertIsNotNone(y1.projected_depreciation)
        self.assertGreater(y1.projected_revenue, 416161.0)

        # Markdown table
        self.assertIn("| Metric ($ Millions) | Base (FY2025) |", forecast_out.forecast_table_markdown)
        self.assertIn("| **Unlevered Free Cash Flow (UFCF)** |", forecast_out.forecast_table_markdown)

        # Rationales
        self.assertTrue(len(forecast_out.growth_rationale) > 20)
        self.assertTrue(len(forecast_out.margin_expansion_rationale) > 10)

    def test_03_force_simplified_mode_override(self):
        """
        Verify force_mode='simplified_nopat_less_capex' override:
        Forces simplified mode even when D&A was provided in audit.
        """
        agent = ForecastingAnalystAgent()
        audit = _create_synthetic_aapl_audit(include_depreciation=True)

        forecast_out = agent.forecast(
            ticker="AAPL",
            fiscal_year=2025,
            financial_audit=audit,
            horizon_years=5,
            force_mode="simplified_nopat_less_capex",
        )

        self.assertEqual(forecast_out.provenance_mode, "simplified_nopat_less_capex")
        y1 = forecast_out.forecast_schedule[0]
        self.assertIsNone(y1.projected_depreciation)

    def test_04_deterministic_cagr_decay_fallback(self):
        """
        Verify deterministic CAGR decay fallback on a non-existent filing:
        - When narrative chunks < 2, fallback triggers deterministically.
        - guidance_source is tagged 'historical_cagr_decay'.
        - Growth rates decay by 75 bps annually towards 2.75% terminal floor.
        """
        agent = ForecastingAnalystAgent()
        audit = _create_synthetic_aapl_audit(include_depreciation=False)

        # Use an un-ingested ticker symbol to guarantee zero narrative chunks retrieved
        forecast_out = agent.forecast(
            ticker="NONEXISTENT_TICKER",
            fiscal_year=2025,
            financial_audit=audit,
            horizon_years=5,
        )

        self.assertEqual(forecast_out.guidance_source, "historical_cagr_decay")
        self.assertEqual(forecast_out.provenance_mode, "simplified_nopat_less_capex")
        self.assertIn("deterministic historical CAGR decay rule", forecast_out.growth_rationale)

        # Verify decay curve properties:
        growths = [y.projected_revenue_growth_pct for y in forecast_out.forecast_schedule]
        self.assertEqual(len(growths), 5)
        for i in range(len(growths) - 1):
            diff = round(growths[i] - growths[i + 1], 2)
            if growths[i + 1] > 2.75:
                self.assertAlmostEqual(diff, 0.75, places=1, msg="Expected ~75 bps annual growth decay")

    def test_05_three_agent_pipeline_audit_to_forecast_to_valuation(self):
        """
        End-to-End 3-Agent Pipeline Test:
        FinancialAuditor -> ForecastingAnalyst -> DCF ValuationSpecialist
        Passes forecast.projected_fcfs directly into valuation engine.
        """
        audit = _create_synthetic_aapl_audit(include_depreciation=True)

        forecaster = ForecastingAnalystAgent()
        forecast_out = forecaster.forecast(
            ticker="AAPL",
            fiscal_year=2025,
            financial_audit=audit,
            horizon_years=5,
        )

        valuation_specialist = ValuationSpecialistAgent()
        val_output = valuation_specialist.value(
            ticker="AAPL",
            fiscal_year=2025,
            financial_audit=audit,
            projected_fcfs=forecast_out.projected_fcfs,
            beta=1.24,
            share_price=230.0,
        )

        self.assertIsInstance(val_output, DCFValuationOutput)
        self.assertEqual(val_output.ticker, "AAPL")
        self.assertEqual(val_output.projected_fcfs, forecast_out.projected_fcfs)
        self.assertGreater(val_output.enterprise_value, 1000000.0)
        self.assertGreater(val_output.implied_fair_value_per_share, 50.0)
        self.assertIn("WACC", val_output.sensitivity_matrix_markdown)
        self.assertTrue(len(val_output.valuation_summary) > 30)


if __name__ == "__main__":
    unittest.main()
