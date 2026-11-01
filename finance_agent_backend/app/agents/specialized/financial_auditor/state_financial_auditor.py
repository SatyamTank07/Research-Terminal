"""State and Output Schemas for Financial Auditor Agent."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.agents.specialized.financial_auditor.findings import ForensicFinding
from app.agents.specialized.financial_auditor.modules.balance_sheet_strength import BalanceSheetResult, SolvencyRatiosResult
from app.agents.specialized.financial_auditor.modules.capital_allocation import YearCapitalIntensity, YearShareCount
from app.agents.specialized.financial_auditor.modules.earnings_normalization import NormalizationAdjustment
from app.agents.specialized.financial_auditor.modules.financial_history import YearFinancialsResult
from app.agents.specialized.financial_auditor.modules.margins_and_costs import YearCostStructure
from app.agents.specialized.financial_auditor.modules.statement_integrity import AuditDataQuality
from app.agents.specialized.financial_auditor.modules.working_capital import YearWorkingCapital

# Ergonomic aliases mapping architectural specification names to calculation engine results
YearFinancials = YearFinancialsResult
BalanceSheetSnapshot = BalanceSheetResult


class ProfitabilityRatiosResult(BaseModel):
    effective_tax_rate_pct: float
    tax_rate_source: str
    equity_status: str
    nopat: float
    invested_capital: float
    capital_basis: Literal["average", "ending"]
    roic_pct: Optional[float] = Field(None, description="NOPAT / (debt + equity - liquid cash)")
    roic_on_gross_capital_pct: Optional[float] = Field(
        None, description="NOPAT / (debt + equity); comparable for cash-rich companies"
    )
    roe_pct: Optional[float] = None
    latest_gross_margin_pct: Optional[float] = None
    latest_operating_margin_pct: float
    latest_net_margin_pct: float


class YearEarningsQuality(BaseModel):
    fiscal_year: int
    accruals: float = Field(..., description="Net income minus operating cash flow")
    sloan_accruals_ratio_pct: Optional[float] = None
    stock_based_compensation: Optional[float] = None
    sbc_to_ocf_pct: Optional[float] = None
    sbc_to_operating_income_pct: Optional[float] = None
    sbc_adjusted_fcf: Optional[float] = None
    non_operating_income: Optional[float] = Field(None, description="Pre-tax income minus operating income")
    non_operating_to_operating_income_pct: Optional[float] = None
    effective_tax_rate_pct: Optional[float] = None
    deferred_tax_to_net_income_pct: Optional[float] = None
    adjustments: List[NormalizationAdjustment] = Field(default_factory=list)
    operating_income_adjustments_total: float = 0.0
    net_income_adjustments_total: float = 0.0
    adjustments_pct_of_operating_income: Optional[float] = None
    normalized_operating_income: float
    normalized_operating_margin_pct: float
    normalized_net_income: float
    normalized_net_margin_pct: float


class FinancialAuditResult(BaseModel):
    """Complete structured output returned by build_audit."""

    multi_year_history: List[YearFinancialsResult]
    balance_sheet: BalanceSheetResult
    profitability_and_return_ratios: ProfitabilityRatiosResult
    solvency_and_liquidity_ratios: SolvencyRatiosResult
    earnings_quality: List[YearEarningsQuality]
    working_capital: List[YearWorkingCapital]
    capital_intensity: List[YearCapitalIntensity]
    cost_structure: List[YearCostStructure]
    share_count_history: List[YearShareCount]
    forensic_findings: List[ForensicFinding]
    rejected_normalization_adjustments: List[Dict[str, Any]] = Field(default_factory=list)


class FinancialAuditOutput(BaseModel):
    """Structured artifact emitted by the Financial Auditor Agent."""

    ticker: str = Field(..., description="Stock ticker symbol")
    fiscal_year: int = Field(..., description="Target 10-K fiscal year audited")

    # 1. Deterministic statements analytics (computed by build_audit)
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
