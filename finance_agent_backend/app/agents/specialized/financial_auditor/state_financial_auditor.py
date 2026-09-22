"""State and Output Schemas for Financial Auditor Agent."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field

from app.agents.tools.financial_math_tools import (
    BalanceSheetResult,
    ProfitabilityRatiosResult,
    SolvencyRatiosResult,
    YearFinancialsResult,
)

# Ergonomic aliases mapping architectural specification names to calculation engine results
YearFinancials = YearFinancialsResult
BalanceSheetSnapshot = BalanceSheetResult


class FinancialAuditOutput(BaseModel):
    """Structured artifact emitted by the Financial Auditor Agent."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="Target 10-K fiscal year audited")

    # 1. Deterministic Math & Multi-Year Ratios (Calculated via tool)
    multi_year_history: List[YearFinancialsResult] = Field(
        ...,
        description="3-year contiguous history (Rev, GP, EBIT, NI, OCF, CapEx, FCF, Margins, YoY Growth)",
    )
    balance_sheet: BalanceSheetResult = Field(
        ...,
        description="Audited liquidity bridge (Cash, Securities, Short/Long Debt, Net Debt, Shares, Equity)",
    )
    profitability_and_return_ratios: ProfitabilityRatiosResult = Field(
        ...,
        description="ROIC, ROE, NOPAT, Invested Capital, and derived effective tax rate",
    )
    solvency_and_liquidity_ratios: SolvencyRatiosResult = Field(
        ...,
        description="Net Debt/EBITDA, Current Ratio, Debt/Equity",
    )
    forensic_red_flags: List[str] = Field(
        default_factory=list,
        description="Algorithmic red flags (accrual divergence, weak FCF conversion, AR/Inventory lag)",
    )
    restatement_notes: List[str] = Field(
        default_factory=list,
        description="Audit notes on cross-filing comparisons, restatements, or reclassifications across filings",
    )

    # 2. Raw Markdown Tables (Preserved for UI presentation & Report Synthesizer)
    income_statement_markdown_table: str = Field(
        default="", description="Audited Statement of Operations Markdown table"
    )
    balance_sheet_markdown_table: str = Field(
        default="", description="Audited Consolidated Balance Sheets Markdown table"
    )
    cash_flow_markdown_table: str = Field(
        default="", description="Audited Statement of Cash Flows Markdown table"
    )

    # 3. Auditor Synthesis & Audit Trail
    auditor_summary: str = Field(
        ...,
        description="Qualitative synthesis explaining earnings quality, working capital dynamics, and red flag context",
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of chunk_ids, breadcrumbs, and table titles used in the audit",
    )
