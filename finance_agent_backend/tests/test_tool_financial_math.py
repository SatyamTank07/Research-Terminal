"""Unit tests for financial_math_tools.py (Milestone 2).

Validates:
1. Deterministic calculation of multi-year profitability margins and Free Cash Flow.
2. Effective tax rate derivation with exact provenance tracking and distinct fallback flags.
3. Consistent liquid cash treatment between Net Debt and ROIC Invested Capital.
4. Net cash surplus representation (raw signed negative number + net_cash_position boolean).
5. Sequenced algorithmic forensic red-flag detection (Tier 1 earnings divergence, Tier 2 AR, Tier 3 Inventory).
6. Real-world 10-K benchmarks across Apple (FY25), Tesla (FY25), and Nvidia (FY26).
7. Strict validation guards on diluted share counts scale (preventing silent corruption).
8. Pydantic AliasChoices flexibility for seamless LLM agent calling.
"""

import unittest
from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    FinancialAuditResult,
    calculate_financial_ratios,
    detect_forensic_red_flags,
    audit_financial_metrics,
    audit_financial_metrics_tool,
)


class TestFinancialMathTools(unittest.TestCase):
    """Test suite for deterministic financial math and forensic audit engine."""

    def test_01_apple_fy2025_audit(self):
        """
        Verify audited calculations and real-world forensic red flag on Apple Inc. FY2025 10-K.
        In FY25, Apple's Net Income grew +19.5% while Operating Cash Flow dropped -5.7%
        due to cash tax payments on prior-year EU accruals.
        """
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
                pretax_income=132729.0,
                income_tax_expense=20719.0,
                net_income=112010.0,
                operating_cash_flow=111482.0,
                capital_expenditures=12715.0,
                depreciation_amortization=11698.0,
            ),
        ]

        balance_sheet = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=35934.0,
            marketable_securities=96486.0,  # 18,763 current + 77,723 non-current
            short_term_debt=20329.0,        # 7,979 commercial paper + 12,350 term debt
            long_term_debt=78328.0,
            stockholders_equity=73733.0,
            diluted_shares_outstanding=15004.7,
            current_assets=147957.0,
            current_liabilities=165631.0,
        )

        result = audit_financial_metrics(annual_financials, balance_sheet)

        # 1. Multi-year history checks
        hist = result["multi_year_history"]
        self.assertEqual(len(hist), 3)

        # FY2025 margins
        fy25 = hist[2]
        self.assertEqual(fy25["fiscal_year"], 2025)
        self.assertAlmostEqual(fy25["gross_margin_pct"], 46.91, places=2)
        self.assertAlmostEqual(fy25["operating_margin_pct"], 31.97, places=2)
        self.assertAlmostEqual(fy25["net_margin_pct"], 26.92, places=2)
        self.assertAlmostEqual(fy25["free_cash_flow"], 98767.0, places=1)  # 111,482 - 12,715
        self.assertAlmostEqual(fy25["revenue_growth_pct"], 6.43, places=2)

        # 2. Tax Rate Derivation (Apple's real effective tax rate ~15.61% vs statutory 21%)
        prof = result["profitability_and_return_ratios"]
        self.assertAlmostEqual(prof["effective_tax_rate_pct"], 15.61, places=1)
        self.assertEqual(prof["tax_rate_source"], "derived_from_10k")

        # 3. Solvency & Liquid Cash Balance Bridge
        bs = result["balance_sheet"]
        self.assertEqual(bs["total_liquid_cash"], 132420.0)  # 35,934 + 96,486
        self.assertEqual(bs["total_debt"], 98657.0)          # 20,329 + 78,328
        self.assertEqual(bs["net_debt"], -33763.0)           # Cash surplus
        self.assertTrue(bs["net_cash_position"])

        solv = result["solvency_and_liquidity_ratios"]
        self.assertEqual(solv["net_debt_to_ebitda_interpretation"], "net_cash_surplus")
        self.assertLess(solv["net_debt_to_ebitda"], 0.0)
        self.assertAlmostEqual(solv["current_ratio"], 0.89, places=2)

        # 4. Forensic Red-Flag Screening (Tier 1 real-world trigger)
        flags = result["forensic_red_flags"]
        self.assertTrue(any("EARNINGS QUALITY DIVERGENCE (FY2025)" in f for f in flags))
        self.assertTrue(any("Net income increased by +19.50%" in f for f in flags))

    def test_02_tesla_fy2025_audit(self):
        """Verify Tesla FY2025 audit calculations with net cash position and derived tax rate."""
        annual_financials = [
            AnnualFinancialInput(
                fiscal_year=2023,
                revenue=96773.0,
                gross_profit=17660.0,
                operating_income=8891.0,
                net_income=14974.0,
                operating_cash_flow=13256.0,
                capital_expenditures=8899.0,
            ),
            AnnualFinancialInput(
                fiscal_year=2024,
                revenue=97690.0,
                gross_profit=17450.0,
                operating_income=7076.0,
                net_income=7153.0,
                operating_cash_flow=14923.0,
                capital_expenditures=11342.0,
            ),
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=94827.0,
                gross_profit=17094.0,
                operating_income=4355.0,
                pretax_income=4493.0,
                income_tax_expense=638.0,
                net_income=3855.0,
                operating_cash_flow=14747.0,
                capital_expenditures=8527.0,
                depreciation_amortization=6148.0,
            ),
        ]

        balance_sheet = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=16513.0,
            marketable_securities=27546.0,
            short_term_debt=1640.0,
            long_term_debt=6060.0,
            stockholders_equity=78540.0,
            diluted_shares_outstanding=3529.0,
            current_assets=68642.0,
            current_liabilities=31714.0,
        )

        result = audit_financial_metrics(annual_financials, balance_sheet)

        # Tax rate derived ~14.20%
        prof = result["profitability_and_return_ratios"]
        self.assertAlmostEqual(prof["effective_tax_rate_pct"], 14.20, places=1)
        self.assertEqual(prof["tax_rate_source"], "derived_from_10k")

        # FCF positive: 14,747 - 8,527 = 6,220
        fy25 = result["multi_year_history"][2]
        self.assertAlmostEqual(fy25["free_cash_flow"], 6220.0, places=1)

        # Net cash position
        bs = result["balance_sheet"]
        self.assertTrue(bs["net_cash_position"])
        self.assertLess(bs["net_debt"], 0.0)

    def test_03_nvidia_fy2026_audit_and_tier3_inventory_flag(self):
        """
        Verify Nvidia FY2026 audit:
        1. High growth (+65.5% revenue) and exceptional ROIC (>100%).
        2. Tier 3 Inventory red flag: Inventory grew +112.3% while revenue grew +65.5%.
        """
        annual_financials = [
            AnnualFinancialInput(
                fiscal_year=2024,
                revenue=60922.0,
                gross_profit=44301.0,
                operating_income=32972.0,
                net_income=29760.0,
                operating_cash_flow=28090.0,
                capital_expenditures=1068.0,
                accounts_receivable=9999.0,
                inventories=5282.0,
            ),
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=130497.0,
                gross_profit=97858.0,
                operating_income=81453.0,
                net_income=72880.0,
                operating_cash_flow=64089.0,
                capital_expenditures=3000.0,
                accounts_receivable=23065.0,
                inventories=10080.0,
            ),
            AnnualFinancialInput(
                fiscal_year=2026,
                revenue=215938.0,
                gross_profit=153463.0,
                operating_income=130387.0,
                pretax_income=141450.0,
                income_tax_expense=21383.0,
                net_income=120067.0,
                operating_cash_flow=102718.0,
                capital_expenditures=4500.0,
                depreciation_amortization=2843.0,
                accounts_receivable=38466.0,
                inventories=21403.0,
            ),
        ]

        balance_sheet = BalanceSheetInput(
            fiscal_year=2026,
            cash_and_equivalents=10605.0,
            marketable_securities=51951.0,
            short_term_debt=999.0,
            long_term_debt=7469.0,
            stockholders_equity=157293.0,
            diluted_shares_outstanding=24514.0,
        )

        result = audit_financial_metrics(annual_financials, balance_sheet)

        # Revenue growth
        fy26 = result["multi_year_history"][2]
        self.assertAlmostEqual(fy26["revenue_growth_pct"], 65.47, places=1)
        self.assertAlmostEqual(fy26["gross_margin_pct"], 71.07, places=1)

        # Derived tax rate ~15.12%
        prof = result["profitability_and_return_ratios"]
        self.assertAlmostEqual(prof["effective_tax_rate_pct"], 15.12, places=1)

        # ROIC > 100%
        self.assertGreater(prof["roic_pct"], 100.0)

        # Net cash surplus of ~$54B
        bs = result["balance_sheet"]
        self.assertEqual(bs["net_debt"], -54088.0)
        self.assertTrue(bs["net_cash_position"])

        # Tier 3 Inventory Red Flag triggered (Inventory grew +112.3% vs Revenue +65.5%)
        flags = result["forensic_red_flags"]
        self.assertTrue(any("INVENTORY ACCUMULATION WARNING (FY2026)" in f for f in flags))
        self.assertTrue(any("substantially exceeding revenue growth" in f for f in flags))

    def test_04_tax_rate_provenance_fallbacks(self):
        """Verify the distinct fallback flags for pre-tax losses, tax credits, and missing data."""
        base_annual = [
            AnnualFinancialInput(
                fiscal_year=2024,
                revenue=100.0,
                gross_profit=40.0,
                operating_income=10.0,
                net_income=8.0,
                operating_cash_flow=12.0,
                capital_expenditures=2.0,
            )
        ]
        base_bs = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=20.0,
            stockholders_equity=50.0,
            diluted_shares_outstanding=10.0,
        )

        # Case A: Pre-tax loss (pretax_income <= 0)
        annual_loss = [
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=100.0,
                gross_profit=20.0,
                operating_income=-15.0,
                pretax_income=-10.0,
                income_tax_expense=2.0,
                net_income=-12.0,
                operating_cash_flow=5.0,
                capital_expenditures=2.0,
            )
        ]
        res_loss = calculate_financial_ratios(annual_loss, base_bs)
        self.assertEqual(res_loss["profitability_and_return_ratios"].effective_tax_rate_pct, 21.0)
        self.assertEqual(res_loss["profitability_and_return_ratios"].tax_rate_source, "statutory_fallback_due_to_pretax_loss")

        # Case B: Tax benefit (income_tax_expense < 0)
        annual_benefit = [
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=100.0,
                gross_profit=50.0,
                operating_income=25.0,
                pretax_income=20.0,
                income_tax_expense=-5.0,  # tax credit/benefit
                net_income=25.0,
                operating_cash_flow=20.0,
                capital_expenditures=3.0,
            )
        ]
        res_benefit = calculate_financial_ratios(annual_benefit, base_bs)
        self.assertEqual(res_benefit["profitability_and_return_ratios"].effective_tax_rate_pct, 21.0)
        self.assertEqual(res_benefit["profitability_and_return_ratios"].tax_rate_source, "statutory_fallback_due_to_tax_benefit")

        # Case C: Missing tax data
        annual_missing = [
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=100.0,
                gross_profit=50.0,
                operating_income=25.0,
                net_income=20.0,
                operating_cash_flow=20.0,
                capital_expenditures=3.0,
            )
        ]
        res_missing = calculate_financial_ratios(annual_missing, base_bs)
        self.assertEqual(res_missing["profitability_and_return_ratios"].effective_tax_rate_pct, 21.0)
        self.assertEqual(res_missing["profitability_and_return_ratios"].tax_rate_source, "statutory_default_not_provided")

    def test_05_share_count_validation_guard(self):
        """Confirm ValueError is raised with clear instructions when shares > 100,000 (thousands/raw units)."""
        with self.assertRaises(ValueError) as ctx:
            BalanceSheetInput(
                fiscal_year=2025,
                cash_and_equivalents=30.0,
                stockholders_equity=50.0,
                diluted_shares_outstanding=15004697.0,  # In thousands instead of millions
            )
        self.assertIn("appears to be passed in thousands or raw units", str(ctx.exception))
        self.assertIn("standardized to MILLIONS", str(ctx.exception))

        # Negative share count guard
        with self.assertRaises(ValueError) as ctx_neg:
            BalanceSheetInput(
                fiscal_year=2025,
                cash_and_equivalents=30.0,
                stockholders_equity=50.0,
                diluted_shares_outstanding=-5.0,
            )
        self.assertIn("must be strictly positive", str(ctx_neg.exception))

    def test_06_alias_flexibility_and_langchain_tool(self):
        """Confirm Pydantic AliasChoices parse synonyms and the LangChain @tool decorator executes cleanly."""
        raw_annual_dict = [
            {
                "year": 2024,
                "net_sales": 1000.0,
                "gross_margin": 450.0,
                "ebit": 250.0,
                "net_earnings": 180.0,
                "cash_from_operations": 220.0,
                "capex": 50.0,  # Enforced positive magnitude
            },
            {
                "year": 2025,
                "total_net_sales": 1200.0,
                "gross_profit": 550.0,
                "income_from_operations": 320.0,
                "net_income": 240.0,
                "cash_provided_by_operating_activities": 300.0,
                "payments_for_acquisition_of_property_plant_and_equipment": 60.0,
            },
        ]
        raw_bs_dict = {
            "year": 2025,
            "cash": 150.0,
            "short_term_investments": 250.0,
            "commercial_paper": 50.0,
            "term_debt": 200.0,
            "shareholders_equity": 600.0,
            "shares_outstanding": 100.0,
        }

        # Validate through LangChain @tool wrapper
        result = audit_financial_metrics_tool.invoke({
            "annual_financials": raw_annual_dict,
            "balance_sheet": raw_bs_dict,
        })

        self.assertIn("multi_year_history", result)
        self.assertEqual(len(result["multi_year_history"]), 2)
        # Verify CapEx calculation: 300 - 60 = 240
        self.assertEqual(result["multi_year_history"][1]["free_cash_flow"], 240.0)
        # Verify net debt: (50 + 200) - (150 + 250) = 250 - 400 = -150.0
        self.assertEqual(result["balance_sheet"]["net_debt"], -150.0)
        self.assertTrue(result["balance_sheet"]["net_cash_position"])

    def test_07_fcf_conversion_chronically_weak_flag(self):
        """
        Verify Bug Fix 1: Flag chronically weak FCF conversion (< 70%).
        Tests a company with 30% and 27.27% FCF conversion across consecutive years.
        """
        annual_financials = [
            AnnualFinancialInput(
                fiscal_year=2024,
                revenue=500.0,
                gross_profit=250.0,
                operating_income=120.0,
                net_income=100.0,
                operating_cash_flow=105.0,  # OCF > NI, but heavy capex
                capital_expenditures=75.0,  # FCF = 30.0 -> FCF conversion = 30.0%
            ),
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=550.0,
                gross_profit=280.0,
                operating_income=130.0,
                net_income=110.0,
                operating_cash_flow=110.0,
                capital_expenditures=80.0,  # FCF = 30.0 -> FCF conversion = 27.27%
            ),
        ]
        balance_sheet = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=50.0,
            stockholders_equity=200.0,
            diluted_shares_outstanding=50.0,
        )

        result = audit_financial_metrics(annual_financials, balance_sheet)
        flags = result["forensic_red_flags"]

        self.assertTrue(
            any("WEAK CASH CONVERSION WARNING" in f for f in flags),
            f"Expected weak cash conversion warning in flags, got: {flags}"
        )
        self.assertTrue(any("chronically weak (< 70%)" in f for f in flags))

    def test_08_negative_stockholders_equity_suppression(self):
        """
        Verify Bug Fix 2: Suppress ROE, ROIC, and Debt/Equity when stockholders' equity < 0.
        Prevents distorted ROIC/ROE and sets explicit equity_status flag and forensic warning.
        """
        annual_financials = [
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=1000.0,
                gross_profit=400.0,
                operating_income=300.0,
                net_income=200.0,
                operating_cash_flow=250.0,
                capital_expenditures=50.0,
            )
        ]
        balance_sheet = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=100.0,
            short_term_debt=50.0,
            long_term_debt=200.0,         # Total debt = 250.0
            stockholders_equity=-100.0,    # Negative equity (-$100M)
            diluted_shares_outstanding=50.0,
        )

        result = audit_financial_metrics(annual_financials, balance_sheet)
        prof = result["profitability_and_return_ratios"]
        solv = result["solvency_and_liquidity_ratios"]
        flags = result["forensic_red_flags"]

        # Ratios must be suppressed to None
        self.assertIsNone(prof["roic_pct"])
        self.assertIsNone(prof["roe_pct"])
        self.assertIsNone(solv["total_debt_to_equity"])
        self.assertEqual(prof["equity_status"], "negative_book_equity_ratios_suppressed")

        # Forensic red flag must be emitted
        self.assertTrue(
            any("NEGATIVE BOOK EQUITY WARNING" in f for f in flags),
            f"Expected negative book equity warning in flags, got: {flags}"
        )

    def test_09_non_contiguous_years_and_temporal_mismatch_guards(self):
        """
        Verify Bug Fix 3: Strict validation guards for contiguous years and temporal matching.
        """
        # Case A: Non-contiguous years (FY2023 and FY2025, skipping FY2024)
        non_contiguous_annuals = [
            AnnualFinancialInput(
                fiscal_year=2023,
                revenue=100.0,
                gross_profit=50.0,
                operating_income=20.0,
                net_income=15.0,
                operating_cash_flow=25.0,
                capital_expenditures=5.0,
            ),
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=200.0,
                gross_profit=100.0,
                operating_income=40.0,
                net_income=30.0,
                operating_cash_flow=50.0,
                capital_expenditures=10.0,
            ),
        ]
        valid_bs = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=20.0,
            stockholders_equity=50.0,
            diluted_shares_outstanding=10.0,
        )

        with self.assertRaises(ValueError) as ctx_gap:
            calculate_financial_ratios(non_contiguous_annuals, valid_bs)
        self.assertIn("contiguous fiscal years with no gaps", str(ctx_gap.exception))

        # Case B: Temporal mismatch (2023 income statement paired with 2025 balance sheet)
        contiguous_annuals = [
            AnnualFinancialInput(
                fiscal_year=2023,
                revenue=100.0,
                gross_profit=50.0,
                operating_income=20.0,
                net_income=15.0,
                operating_cash_flow=25.0,
                capital_expenditures=5.0,
            )
        ]
        mismatched_bs = BalanceSheetInput(
            fiscal_year=2025,  # Mismatched year
            cash_and_equivalents=20.0,
            stockholders_equity=50.0,
            diluted_shares_outstanding=10.0,
        )

        with self.assertRaises(ValueError) as ctx_mismatch:
            calculate_financial_ratios(contiguous_annuals, mismatched_bs)
        self.assertIn("must match the latest annual_financials fiscal year", str(ctx_mismatch.exception))

    def test_10_negative_capex_validation_guard(self):
        """
        Verify Bug Fix 4: Reject negative CapEx values to enforce consistency with positive magnitude policy.
        """
        with self.assertRaises(ValueError) as ctx_capex:
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=100.0,
                gross_profit=50.0,
                operating_income=20.0,
                net_income=15.0,
                operating_cash_flow=25.0,
                capital_expenditures=-12715.0,  # Negative CapEx rejected
            )
        self.assertIn("must be passed as a positive magnitude", str(ctx_capex.exception))

    def test_11_current_liabilities_zero_handling(self):
        """
        Verify Bug Fix 5: Explicit is not None check handles current_liabilities=0.0 cleanly.
        """
        annuals = [
            AnnualFinancialInput(
                fiscal_year=2025,
                revenue=100.0,
                gross_profit=50.0,
                operating_income=20.0,
                net_income=15.0,
                operating_cash_flow=25.0,
                capital_expenditures=5.0,
            )
        ]
        bs_zero_liab = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=20.0,
            stockholders_equity=50.0,
            diluted_shares_outstanding=10.0,
            current_assets=50.0,
            current_liabilities=0.0,  # Zero liabilities
        )
        res = calculate_financial_ratios(annuals, bs_zero_liab)
        # Should be None (not divide by zero or raise unhandled exception)
        self.assertIsNone(res["solvency_and_liquidity_ratios"].current_ratio)

    def test_12_negative_balance_sheet_fields_guard(self):
        """
        Verify that negative current_liabilities, current_assets, cash, or debt
        raise a ValidationError rather than silently producing garbage or masking extraction errors.
        """
        from pydantic import ValidationError

        # Negative current_liabilities rejected
        with self.assertRaises(ValidationError):
            BalanceSheetInput(
                fiscal_year=2025,
                cash_and_equivalents=20.0,
                stockholders_equity=50.0,
                diluted_shares_outstanding=10.0,
                current_liabilities=-50.0,
            )

        # Negative cash_and_equivalents rejected
        with self.assertRaises(ValidationError):
            BalanceSheetInput(
                fiscal_year=2025,
                cash_and_equivalents=-10.0,
                stockholders_equity=50.0,
                diluted_shares_outstanding=10.0,
            )

        # Negative short_term_debt rejected
        with self.assertRaises(ValidationError):
            BalanceSheetInput(
                fiscal_year=2025,
                cash_and_equivalents=20.0,
                short_term_debt=-5.0,
                stockholders_equity=50.0,
                diluted_shares_outstanding=10.0,
            )

        # But stockholders_equity CAN be legitimately negative
        valid_neg_equity_bs = BalanceSheetInput(
            fiscal_year=2025,
            cash_and_equivalents=20.0,
            stockholders_equity=-50.0,
            diluted_shares_outstanding=10.0,
        )
        self.assertEqual(valid_neg_equity_bs.stockholders_equity, -50.0)


if __name__ == "__main__":
    unittest.main()
