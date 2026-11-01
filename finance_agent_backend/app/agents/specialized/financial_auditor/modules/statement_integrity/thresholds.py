"""Tolerances for trusting the extracted statements."""

from app.agents.specialized.financial_auditor.context.extract import REQUIRED_ANNUAL_FIELDS

IDENTITY_TOLERANCE_PCT = 0.5

# Fields that an institutional audit expects to find; drives the completeness score.
EXPECTED_LATEST_FIELDS = REQUIRED_ANNUAL_FIELDS + [
    "gross_profit_or_cost_of_revenue",
    "pretax_income",
    "income_tax_expense",
    "depreciation_amortization",
    "stock_based_compensation",
    "diluted_weighted_shares",
    "current_assets",
    "current_liabilities",
    "total_assets",
]
