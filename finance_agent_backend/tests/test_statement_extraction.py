"""Unit tests for deterministic statement parsing and line-item extraction.

Validates:
1. Legacy `Col_n` markdown (split `$` cells, colspan drift) parses into aligned fiscal periods.
2. TSLA FY2025 line items, including the fields the LLM previously dropped
   (short-term investments, gross profit, D&A, SBC, pretax/tax, current assets/liabilities).
3. Share counts: weighted diluted (income statement) and period-end (equity caption), thousands -> millions.
4. Debt classification around the "Total current liabilities" subtotal.
5. Accounting identity checks and Latest Filing Precedence cross-filing differences.
6. A second filer vocabulary (AAPL-style) to guard against overfitting.
7. Clean statement rendering.
"""

import unittest

from app.agents.tools.financial_math_tools import run_audit_from_inputs
from app.agents.tools.statement_extraction import build_statement_grids, extract_audit_inputs
from app.services.financial_tables import (
    is_well_formed,
    parse_statement_rows,
    parse_statement_table,
    parse_value_token,
    render_clean_statement,
)
from tests.statement_fixtures import (
    TSLA_FY2025_BALANCE_SHEET,
    TSLA_FY2025_INCOME_STATEMENT,
    aapl_payloads,
    tsla_payloads,
)


class TestFinancialTableParsing(unittest.TestCase):
    def test_01_value_tokens(self):
        self.assertEqual(parse_value_token("( 5,001 )"), -5001.0)
        self.assertEqual(parse_value_token("—"), 0.0)
        self.assertIsNone(parse_value_token("$"))
        self.assertIsNone(parse_value_token(""))
        self.assertEqual(parse_value_token("$3,794"), 3794.0)
        self.assertEqual(parse_value_token("1.08"), 1.08)

    def test_02_legacy_income_statement_periods_and_alignment(self):
        grid = parse_statement_table(TSLA_FY2025_INCOME_STATEMENT)
        self.assertEqual(grid.periods, [2025, 2024, 2023])
        self.assertTrue(is_well_formed(grid))
        rows = {}
        for r in grid.value_rows:
            rows.setdefault(r.norm_label, r)  # first occurrence (revenue section, not cost section)
        self.assertEqual(rows["automotive sales"].values, {2025: 65821.0, 2024: 72480.0, 2023: 78509.0})
        self.assertEqual(rows["automotive sales"].section, "revenues")
        self.assertEqual(rows["automotive regulatory credits"].values[2024], 2763.0)
        self.assertEqual(rows["interest expense"].values[2025], -338.0)
        self.assertEqual(rows["restructuring and other"].values[2023], 0.0)

    def test_03_balance_sheet_date_headers(self):
        grid = parse_statement_table(TSLA_FY2025_BALANCE_SHEET)
        self.assertEqual(grid.periods, [2025, 2024])

    def test_04_clean_rendering_round_trip(self):
        grid = parse_statement_table(TSLA_FY2025_INCOME_STATEMENT)
        md = render_clean_statement(grid)
        self.assertIn("| Line item | FY2025 | FY2024 | FY2023 |", md)
        self.assertIn("| Total revenues | 94,827 | 97,690 | 96,773 |", md)
        self.assertIn("| Provision for (benefit from) income taxes | 1,423 | 1,837 | (5,001) |", md)
        self.assertNotIn("Col_", md)
        reparsed = parse_statement_table(md)
        self.assertEqual(reparsed.periods, [2025, 2024, 2023])
        self.assertEqual(
            {r.norm_label: r.values for r in reparsed.value_rows}["total revenues"],
            {2025: 94827.0, 2024: 97690.0, 2023: 96773.0},
        )

    def test_05_raw_grid_rows_like_ingestion(self):
        rows = [
            ["", "2025", "", "2024"],
            ["Total revenues", "$", "100", "$ 90"],
            ["Net income", "(5", ")", "7"],
        ]
        grid = parse_statement_rows(rows)
        self.assertEqual(grid.periods, [2025, 2024])
        vals = {r.norm_label: r.values for r in grid.value_rows}
        self.assertEqual(vals["total revenues"], {2025: 100.0, 2024: 90.0})
        self.assertEqual(vals["net income"], {2025: -5.0, 2024: 7.0})


class TestTeslaExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = extract_audit_inputs(build_statement_grids(tsla_payloads()), 2025, [2023, 2024, 2025])

    def test_01_usable_window(self):
        self.assertTrue(self.result.is_usable)
        self.assertEqual(self.result.years, [2023, 2024, 2025])
        self.assertEqual(self.result.missing_fields, [])
        self.assertEqual(self.result.completeness_score, 1.0)

    def test_02_income_statement_fields(self):
        fy25 = self.result.annual[2025]
        self.assertEqual(fy25["revenue"], 94827.0)
        self.assertEqual(fy25["cost_of_revenue"], 77733.0)
        self.assertEqual(fy25["gross_profit"], 17094.0)
        self.assertEqual(fy25["operating_expenses"], 12739.0)
        self.assertEqual(fy25["operating_income"], 4355.0)
        self.assertEqual(fy25["pretax_income"], 5278.0)
        self.assertEqual(fy25["income_tax_expense"], 1423.0)
        self.assertEqual(fy25["net_income_total"], 3855.0)
        self.assertEqual(fy25["net_income"], 3794.0)  # attributable to common stockholders
        self.assertEqual(fy25["diluted_weighted_shares"], 3528.0)
        self.assertEqual(self.result.annual[2023]["income_tax_expense"], -5001.0)
        self.assertEqual(self.result.annual[2023]["net_income"], 14997.0)

    def test_03_cash_flow_fields(self):
        fy25 = self.result.annual[2025]
        self.assertEqual(fy25["depreciation_amortization"], 6148.0)
        self.assertEqual(fy25["stock_based_compensation"], 2825.0)
        self.assertEqual(fy25["deferred_income_taxes"], 123.0)
        self.assertEqual(fy25["operating_cash_flow"], 14747.0)
        self.assertEqual(fy25["capital_expenditures"], 8527.0)
        self.assertEqual(self.result.annual[2024]["capital_expenditures"], 11342.0)

    def test_04_balance_sheet_fields(self):
        bs = self.result.to_balance_sheet_input()
        self.assertEqual(bs["cash_and_equivalents"], 16513.0)
        self.assertEqual(bs["marketable_securities"], 27546.0)
        self.assertEqual(bs["short_term_debt"], 1640.0)
        self.assertEqual(bs["long_term_debt"], 6736.0)
        self.assertEqual(bs["stockholders_equity"], 82137.0)
        self.assertEqual(bs["current_assets"], 68642.0)
        self.assertEqual(bs["current_liabilities"], 31714.0)
        self.assertEqual(bs["total_assets"], 137806.0)
        self.assertEqual(bs["weighted_diluted_shares"], 3528.0)
        self.assertEqual(bs["period_end_shares_outstanding"], 3751.0)
        self.assertEqual(bs["noncontrolling_interests"], 728.0)  # 670 equity NCI + 58 redeemable

        prior = self.result.to_prior_balance_sheet_input()
        self.assertEqual(prior["fiscal_year"], 2024)
        self.assertEqual(prior["marketable_securities"], 20424.0)
        self.assertEqual(prior["stockholders_equity"], 72913.0)

        annual = {r["fiscal_year"]: r for r in self.result.to_annual_inputs()}
        self.assertEqual(annual[2024]["period_end_shares_outstanding"], 3216.0)
        self.assertEqual(annual[2025]["accounts_payable"], 13371.0)
        self.assertNotIn("accounts_receivable", annual[2023])

    def test_05_identity_checks_pass(self):
        self.assertTrue(self.result.identity_checks)
        self.assertEqual(self.result.failed_identity_checks, [])

    def test_06_baseline_audit_matches_filing(self):
        audit = run_audit_from_inputs(self.result.to_tool_kwargs())
        bs = audit["balance_sheet"]
        self.assertEqual(bs["total_liquid_cash"], 44059.0)
        self.assertEqual(bs["net_debt"], -35683.0)
        self.assertEqual(bs["valuation_shares_outstanding"], 3751.0)
        self.assertEqual(bs["share_count_source"], "period_end_basic_exceeds_weighted_diluted")
        prof = audit["profitability_and_return_ratios"]
        self.assertAlmostEqual(prof["effective_tax_rate_pct"], 26.96, places=2)
        self.assertEqual(prof["capital_basis"], "average")
        self.assertAlmostEqual(prof["roic_pct"], 6.99, places=2)
        self.assertAlmostEqual(prof["roe_pct"], 4.89, places=2)
        solv = audit["solvency_and_liquidity_ratios"]
        self.assertEqual(solv["ebitda"], 10503.0)
        self.assertAlmostEqual(solv["net_debt_to_ebitda"], -3.40, places=2)
        self.assertAlmostEqual(solv["current_ratio"], 2.16, places=2)
        self.assertAlmostEqual(audit["multi_year_history"][-1]["gross_margin_pct"], 18.03, places=2)

    def test_07_clean_tables_and_citations(self):
        self.assertIn("| Total revenues | 94,827 | 97,690 | 96,773 |", self.result.clean_tables["income_statement"])
        self.assertIn("FY2025", self.result.clean_tables["balance_sheet"])
        self.assertEqual({c["chunk_id"] for c in self.result.citations}, {"tsla-is", "tsla-bs", "tsla-cf"})


class TestCrossFilingAndIdentityFailures(unittest.TestCase):
    def test_01_latest_filing_precedence_and_restatement_candidates(self):
        payloads = tsla_payloads()
        older_is = TSLA_FY2025_INCOME_STATEMENT.replace("| 2025 | Col_3 | 2024 | Col_5 | 2023 |",
                                                        "| 2024 | Col_3 | 2023 | Col_5 | 2022 |")
        # Prior filing reported FY2024 revenue differently (simulated reclassification).
        older_is = older_is.replace("| Total revenues | 94,827 |  |  | 97,690 |", "| Total revenues | 97,000 |  |  | 97,690 |")
        payloads["income_statement"].append({"chunk_id": "tsla-is-2024", "fiscal_year": 2024,
                                             "statement_type": "income_statement", "table_markdown": older_is})
        result = extract_audit_inputs(build_statement_grids(payloads), 2025, [2023, 2024, 2025])
        self.assertEqual(result.annual[2024]["revenue"], 97690.0)  # latest filing wins
        diffs = [d for d in result.cross_filing_differences if d["field"] == "revenue" and d["fiscal_year"] == 2024]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["older_value"], 97000.0)
        self.assertEqual(diffs[0]["older_filing_year"], 2024)

    def test_02_balance_sheet_identity_failure_blocks_deterministic_mode(self):
        payloads = tsla_payloads()
        payloads["balance_sheet"][0]["table_markdown"] = TSLA_FY2025_BALANCE_SHEET.replace(
            "| Total liabilities and equity | $ | 137,806 |", "| Total liabilities and equity | $ | 147,806 |"
        )
        result = extract_audit_inputs(build_statement_grids(payloads), 2025, [2023, 2024, 2025])
        self.assertTrue(any(c["critical"] for c in result.failed_identity_checks))
        self.assertFalse(result.is_usable)

    def test_03_missing_required_row_is_reported(self):
        payloads = tsla_payloads()
        payloads["cash_flow"] = []
        result = extract_audit_inputs(build_statement_grids(payloads), 2025, [2023, 2024, 2025])
        self.assertFalse(result.is_usable)
        self.assertIn("FY2025.operating_cash_flow", result.missing_fields)


AMZN_STYLE_INCOME = """| Col_1 | 2023 | Col_3 | 2024 | Col_5 | 2025 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Total net sales | 574,785 |  | 637,959 |  | 716,924 |
| Operating income | 36,852 |  | 68,593 |  | 79,975 |
| Income (loss) before income taxes | 37,557 |  | 68,614 |  | 97,311 |
| Provision for income taxes | ( 7,120 ) |  | ( 9,265 ) |  | ( 19,087 ) |
| Equity-method investment activity, net of tax | ( 12 ) |  | ( 101 ) |  | ( 554 ) |
| Net income | 30,425 |  | 59,248 |  | 77,670 |
| Weighted-average shares used in computation of earnings per share: |  |  |  |  |  |
| Diluted | 10,492 |  | 10,721 |  | 10,827 |"""

AMZN_STYLE_BALANCE = """| Col_1 | December 31, 2024 | Col_3 | December 31, 2025 |
| :--- | :--- | :--- | :--- |
| ASSETS |  |  |  |
| Cash and cash equivalents | 78,779 |  | 86,810 |
| Marketable securities | 22,423 |  | 36,219 |
| Total current assets | 190,867 |  | 229,083 |
| Total assets | 624,894 |  | 818,042 |
| LIABILITIES AND STOCKHOLDERS’ EQUITY |  |  |  |
| Accounts payable | 94,363 |  | 121,909 |
| Total current liabilities | 179,431 |  | 218,005 |
| Long-term lease liabilities | 78,277 |  | 87,339 |
| Long-term debt | 52,623 |  | 65,648 |
| Other long-term liabilities | 28,593 |  | 35,985 |
| Commitments and contingencies (Note 7) |  |  |  |
| Stockholders’ equity |  |  |  |
| Common stock ($0.01 par value; 100,000 shares authorized; 11,108 and 11,246 shares issued; 10,593 and 10,731 shares outstanding) | 111 |  | 112 |
| Total stockholders’ equity | 285,970 |  | 411,065 |
| Total liabilities and stockholders’ equity | 624,894 |  | 818,042 |"""

AMZN_STYLE_CASH_FLOW = """| Col_1 | 2023 | Col_3 | 2024 | Col_5 | 2025 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Net income | 30,425 |  | 59,248 |  | 77,670 |
| Net cash provided by (used in) operating activities | 84,946 |  | 115,877 |  | 139,514 |
| Purchases of property and equipment | ( 52,729 ) |  | ( 83,004 ) |  | ( 128,320 ) |"""


class TestLayoutVariants(unittest.TestCase):
    """Ascending period columns, benefit-positive tax presentation, no 'Total liabilities' subtotal."""

    @classmethod
    def setUpClass(cls):
        payloads = {
            "income_statement": [{"chunk_id": "is", "fiscal_year": 2025, "table_markdown": AMZN_STYLE_INCOME}],
            "balance_sheet": [{"chunk_id": "bs", "fiscal_year": 2025, "table_markdown": AMZN_STYLE_BALANCE}],
            "cash_flow": [{"chunk_id": "cf", "fiscal_year": 2025, "table_markdown": AMZN_STYLE_CASH_FLOW}],
        }
        cls.result = extract_audit_inputs(build_statement_grids(payloads), 2025, [2023, 2024, 2025])

    def test_01_tax_sign_oriented_by_identity(self):
        self.assertEqual(self.result.annual[2025]["income_tax_expense"], 19087.0)
        self.assertEqual(self.result.annual[2023]["income_tax_expense"], 7120.0)
        self.assertTrue(any("benefit-positive" in w for w in self.result.warnings))
        audit = run_audit_from_inputs(self.result.to_tool_kwargs())
        self.assertAlmostEqual(audit["profitability_and_return_ratios"]["effective_tax_rate_pct"], 19.61, places=2)
        self.assertNotIn("TAX_ANOMALY", {f["code"] for f in audit["forensic_findings"]})

    def test_02_debt_without_total_liabilities_row(self):
        bs = self.result.to_balance_sheet_input()
        self.assertEqual(bs["short_term_debt"], 0.0)
        self.assertEqual(bs["long_term_debt"], 65648.0)  # lease liabilities are not debt
        self.assertEqual(bs["marketable_securities"], 36219.0)
        self.assertEqual(bs["period_end_shares_outstanding"], 10731.0)  # ascending columns -> FY2025 second
        self.assertEqual(self.result.to_prior_balance_sheet_input()["long_term_debt"], 52623.0)


class TestSecondFilerVocabulary(unittest.TestCase):
    def test_01_aapl_style_labels_and_thousand_share_scale(self):
        result = extract_audit_inputs(build_statement_grids(aapl_payloads()), 2025, [2023, 2024, 2025])
        self.assertTrue(result.is_usable, result.summary_for_prompt())
        fy25 = result.annual[2025]
        self.assertEqual(fy25["revenue"], 416161.0)
        self.assertEqual(fy25["gross_profit"], 195201.0)
        self.assertEqual(fy25["cost_of_revenue"], 220960.0)
        self.assertEqual(fy25["pretax_income"], 132729.0)
        self.assertEqual(fy25["income_tax_expense"], 20719.0)
        self.assertEqual(fy25["net_income"], 112010.0)
        self.assertEqual(fy25["stock_based_compensation"], 12863.0)
        self.assertEqual(fy25["capital_expenditures"], 12715.0)
        self.assertAlmostEqual(fy25["diluted_weighted_shares"], 15004.697, places=3)

        bs = result.to_balance_sheet_input()
        self.assertEqual(bs["marketable_securities"], 96486.0)  # current + non-current
        self.assertEqual(bs["short_term_debt"], 20329.0)        # commercial paper + current term debt
        self.assertEqual(bs["long_term_debt"], 78328.0)
        self.assertEqual(bs["stockholders_equity"], 73733.0)
        self.assertAlmostEqual(bs["period_end_shares_outstanding"], 14773.26, places=2)
        self.assertEqual(result.failed_identity_checks, [])

        audit = run_audit_from_inputs(result.to_tool_kwargs())
        self.assertEqual(audit["balance_sheet"]["net_debt"], -33763.0)
        self.assertEqual(audit["balance_sheet"]["share_count_source"], "weighted_average_diluted")


if __name__ == "__main__":
    unittest.main()
