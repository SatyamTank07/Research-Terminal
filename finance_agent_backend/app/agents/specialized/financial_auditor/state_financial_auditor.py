"""State and Output Schemas for Financial Auditor Agent."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.agents.specialized.financial_auditor.findings import ForensicFinding
from app.agents.specialized.financial_auditor.modules.financial_history import YearFinancialsResult
from app.agents.specialized.financial_auditor.modules.margins_and_costs import YearCostStructure
from app.agents.specialized.financial_auditor.modules.statement_integrity import AuditDataQuality
from app.agents.tools.financial_math_tools import (
    BalanceSheetResult,
    ProfitabilityRatiosResult,
    SolvencyRatiosResult,
    YearCapitalIntensity,
    YearEarningsQuality,
    YearShareCount,
    YearWorkingCapital,
)

# Ergonomic aliases mapping architectural specification names to calculation engine results
YearFinancials = YearFinancialsResult
BalanceSheetSnapshot = BalanceSheetResult

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
