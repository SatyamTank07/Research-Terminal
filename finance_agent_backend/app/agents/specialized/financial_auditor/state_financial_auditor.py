"""State and Output Schemas for Financial Auditor Agent."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.agents.tools.financial_math_tools import (
    BalanceSheetResult,
    ForensicFinding,
    ProfitabilityRatiosResult,
    SolvencyRatiosResult,
    YearCapitalIntensity,
    YearCostStructure,
    YearEarningsQuality,
    YearFinancialsResult,
    YearShareCount,
    YearWorkingCapital,
)

# Ergonomic aliases mapping architectural specification names to calculation engine results
YearFinancials = YearFinancialsResult
BalanceSheetSnapshot = BalanceSheetResult

ExtractionMode = Literal["deterministic", "llm_fallback"]


class AuditDataQuality(BaseModel):
    """Provenance and integrity diagnostics for the numbers in the audit."""

    extraction_mode: ExtractionMode = Field(
        ..., description="deterministic = Python-parsed statements; llm_fallback = LLM-transcribed inputs"
    )
    completeness_score: Optional[float] = Field(
        None, description="Share of institutionally expected latest-year line items that were found (0-1)"
    )
    missing_fields: List[str] = Field(default_factory=list)
    identity_checks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Accounting identity checks (balance sheet, gross profit, NI tie-out)"
    )
    cross_filing_differences: List[Dict[str, Any]] = Field(
        default_factory=list, description="Same field/year reported differently by consecutive filings"
    )
    reconciliation_log: List[str] = Field(
        default_factory=list, description="Every correction, gap-fill or override applied to extracted inputs"
    )
    field_provenance: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="FY<year>.<field> -> source statement, row labels, chunk_id"
    )
    warnings: List[str] = Field(default_factory=list)
    summary_may_be_stale: bool = Field(
        False, description="True when inputs were corrected after the LLM wrote the auditor summary"
    )


class FinancialAuditOutput(BaseModel):
    """Structured artifact emitted by the Financial Auditor Agent."""

    ticker: str = Field(..., description="Stock ticker symbol")
    fiscal_year: int = Field(..., description="Target 10-K fiscal year audited")

    # 1. Deterministic statements analytics (computed by audit_financial_metrics)
    multi_year_history: List[YearFinancialsResult]
    balance_sheet: BalanceSheetResult
    profitability_and_return_ratios: ProfitabilityRatiosResult
    solvency_and_liquidity_ratios: SolvencyRatiosResult
    earnings_quality: List[YearEarningsQuality]
    working_capital: List[YearWorkingCapital]
    capital_intensity: List[YearCapitalIntensity]
    cost_structure: List[YearCostStructure]
    share_count_history: List[YearShareCount]
    forensic_findings: List[ForensicFinding] = Field(
        ..., description="Severity-ranked algorithmic findings (high -> info)"
    )

    # 2. Cross-filing review & data integrity
    restatement_notes: List[str] = Field(default_factory=list)
    data_quality: AuditDataQuality

    # 3. Normalized statement tables for presentation
    income_statement_markdown_table: str = ""
    balance_sheet_markdown_table: str = ""
    cash_flow_markdown_table: str = ""

    # 4. Auditor narrative & audit trail
    auditor_summary: str = Field(
        ..., description="Five labelled paragraphs, every claim quantified from the tool output"
    )
    citations: List[Dict[str, Any]] = Field(default_factory=list)
