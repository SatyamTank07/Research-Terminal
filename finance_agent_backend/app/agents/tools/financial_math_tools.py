"""Deterministic Financial Math & Forensic Audit Tools.

Enforces Architectural Principle #2: Zero Mathematical Hallucination.

Computes, from verified 10-K line items:
- Multi-year profitability, cash flow and growth history.
- Liquidity bridge, solvency and capital returns (ROIC / ROE on average capital when the
  prior-year balance sheet is supplied).
- Institutional quality-of-earnings analytics: accruals, SBC intensity, non-operating
  income dependence, tax distortions and LLM-identified normalization adjustments.
- Working-capital days, capital intensity / FCF bridge, cost structure and share-count drift.
- Severity-ranked forensic findings.

The LLM only chooses inputs and normalization items; every number is produced here.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agents.specialized.financial_auditor.context.audit_inputs import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
    check_audit_window,
    validate_audit_inputs,
)
from app.agents.specialized.financial_auditor.findings import ForensicFinding, rank_findings
from app.agents.specialized.financial_auditor.modules.balance_sheet_strength import (
    BalanceSheetResult,
    SolvencyRatiosResult,
    classify_book_equity,
    measure_balance_sheet,
    measure_solvency,
    screen_balance_sheet_strength,
)
from app.agents.specialized.financial_auditor.modules.capital_allocation import (
    YearCapitalIntensity,
    YearShareCount,
    measure_capital_intensity,
    measure_capital_returns,
    measure_share_count,
    screen_capital_allocation,
)
from app.agents.specialized.financial_auditor.modules.cash_conversion import measure_cash_conversion, screen_cash_conversion
from app.agents.specialized.financial_auditor.modules.earnings_normalization import (
    NormalizationAdjustment,
    normalize_earnings,
    screen_earnings_normalization,
    screen_normalization_adjustments,
)
from app.agents.specialized.financial_auditor.modules.revenue_quality import screen_revenue_quality
from app.agents.specialized.financial_auditor.modules.tax_quality import (
    latest_tax_rate,
    measure_tax_quality,
    screen_tax_quality,
)
from app.agents.specialized.financial_auditor.modules.working_capital import (
    YearWorkingCapital,
    measure_working_capital,
    screen_working_capital,
)
from app.agents.specialized.financial_auditor.modules.margins_and_costs import (
    YearCostStructure,
    measure_cost_structure,
    screen_margins_and_costs,
)
from app.agents.specialized.financial_auditor.modules.financial_history import (
    YearFinancialsResult,
    build_financial_history,
)

logger = logging.getLogger("finance_agent.tools.financial_math")


# Forensic thresholds (institutional quality-of-earnings screens)



# ==============================================================================
# 2. Output Schemas
# ==============================================================================
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
    """Complete structured output returned by audit_financial_metrics."""

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


# ==============================================================================
# 3. Helpers
# ==============================================================================
# ==============================================================================
# 4. Core Ratio Engine
# ==============================================================================
def calculate_financial_ratios(
    annual_financials: List[AnnualFinancialInput],
    balance_sheet: BalanceSheetInput,
    prior_balance_sheet: Optional[PriorBalanceSheetInput] = None,
) -> Dict[str, Any]:
    """Multi-year history, liquidity bridge, returns and solvency from verified line items."""
    check_audit_window(annual_financials, balance_sheet, prior_balance_sheet)

    sorted_years = sorted(annual_financials, key=lambda x: x.fiscal_year)
    history = build_financial_history(annual_financials)

    latest_input = sorted_years[-1]
    latest = history[-1]

    latest_tax = latest_tax_rate(latest_input)
    balance = measure_balance_sheet(balance_sheet, prior_balance_sheet)
    equity_status = classify_book_equity(balance_sheet.stockholders_equity)
    returns = measure_capital_returns(latest, balance, prior_balance_sheet, latest_tax.rate, equity_status)

    return {
        "multi_year_history": history,
        "balance_sheet": balance,
        "profitability_and_return_ratios": ProfitabilityRatiosResult(
            effective_tax_rate_pct=round(latest_tax.rate * 100.0, 2),
            tax_rate_source=latest_tax.source,
            equity_status=equity_status,
            nopat=returns.nopat,
            invested_capital=returns.invested_capital,
            capital_basis=returns.capital_basis,
            roic_pct=returns.roic_pct,
            roic_on_gross_capital_pct=returns.roic_on_gross_capital_pct,
            roe_pct=returns.roe_pct,
            latest_gross_margin_pct=latest.gross_margin_pct,
            latest_operating_margin_pct=latest.operating_margin_pct,
            latest_net_margin_pct=latest.net_margin_pct,
        ),
        "solvency_and_liquidity_ratios": measure_solvency(balance, latest, latest_input, equity_status),
    }

# ==============================================================================
# 5. Quality-of-Earnings & Operating Analytics
# ==============================================================================
def calculate_quality_of_earnings(
    annual_financials: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
    adjustments: Optional[List[NormalizationAdjustment]] = None,
    prior_total_assets: Optional[float] = None,
) -> Dict[str, List[BaseModel]]:
    """Earnings quality, working capital, capital intensity, cost structure and share-count analytics."""
    inputs = sorted(annual_financials, key=lambda x: x.fiscal_year)
    normalized_by_year = {r.fiscal_year: r for r in normalize_earnings(annual_financials, history, adjustments)}
    cash_by_year = {r.fiscal_year: r for r in measure_cash_conversion(annual_financials, history, prior_total_assets)}
    tax_by_year = {r.fiscal_year: r for r in measure_tax_quality(annual_financials)}
    eq: List[YearEarningsQuality] = []

    for f in inputs:
        cc, tx, nz = cash_by_year[f.fiscal_year], tax_by_year[f.fiscal_year], normalized_by_year[f.fiscal_year]

        eq.append(YearEarningsQuality(
            fiscal_year=f.fiscal_year,
            accruals=cc.accruals,
            sloan_accruals_ratio_pct=cc.sloan_accruals_ratio_pct,
            stock_based_compensation=cc.stock_based_compensation,
            sbc_to_ocf_pct=cc.sbc_to_ocf_pct,
            sbc_to_operating_income_pct=cc.sbc_to_operating_income_pct,
            sbc_adjusted_fcf=cc.sbc_adjusted_fcf,
            non_operating_income=nz.non_operating_income,
            non_operating_to_operating_income_pct=nz.non_operating_to_operating_income_pct,
            effective_tax_rate_pct=tx.effective_tax_rate_pct,
            deferred_tax_to_net_income_pct=tx.deferred_tax_to_net_income_pct,
            adjustments=nz.adjustments,
            operating_income_adjustments_total=nz.operating_income_adjustments_total,
            net_income_adjustments_total=nz.net_income_adjustments_total,
            adjustments_pct_of_operating_income=nz.adjustments_pct_of_operating_income,
            normalized_operating_income=nz.normalized_operating_income,
            normalized_operating_margin_pct=nz.normalized_operating_margin_pct,
            normalized_net_income=nz.normalized_net_income,
            normalized_net_margin_pct=nz.normalized_net_margin_pct,
        ))

    return {
        "earnings_quality": eq,
        "working_capital": measure_working_capital(annual_financials),
        "capital_intensity": measure_capital_intensity(annual_financials, history),
        "cost_structure": measure_cost_structure(annual_financials, history),
        "share_count_history": measure_share_count(annual_financials),
    }


# ==============================================================================
# 6. Forensic Findings Engine
# ==============================================================================
def detect_forensic_findings(
    annual_inputs: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
    analytics: Dict[str, List[BaseModel]],
    balance_sheet: Optional[BalanceSheetInput] = None,
) -> List[ForensicFinding]:
    """Severity-ranked institutional quality-of-earnings and forensic accounting screens."""
    findings: List[ForensicFinding] = []

    def add(code, severity, year, metric, value, threshold, message):
        findings.append(ForensicFinding(
            code=code, severity=severity, fiscal_year=year, metric=metric,
            value=round(value, 2) if value is not None else None, threshold=threshold, message=message,
        ))

    findings.extend(screen_cash_conversion(history, analytics["earnings_quality"]))
    findings.extend(screen_tax_quality(annual_inputs, analytics["earnings_quality"]))
    findings.extend(screen_earnings_normalization(analytics["earnings_quality"]))
    findings.extend(screen_margins_and_costs(history, analytics["cost_structure"]))
    findings.extend(screen_working_capital(annual_inputs, history, analytics["working_capital"]))
    findings.extend(screen_revenue_quality(annual_inputs, history))
    findings.extend(screen_balance_sheet_strength(balance_sheet))
    findings.extend(screen_capital_allocation(
        history, analytics["capital_intensity"], analytics["share_count_history"]))
    return rank_findings(findings)


# ==============================================================================
# 7. Audit Coordinator & LangChain Tool
# ==============================================================================
def audit_financial_metrics(
    annual_financials: List[AnnualFinancialInput],
    balance_sheet: BalanceSheetInput,
    prior_balance_sheet: Optional[PriorBalanceSheetInput] = None,
    normalization_adjustments: Optional[List[NormalizationAdjustment]] = None,
) -> Dict[str, Any]:
    """Runs ratios, quality-of-earnings analytics and forensic screens; returns a FinancialAuditResult dict."""
    calc = calculate_financial_ratios(annual_financials, balance_sheet, prior_balance_sheet)
    screened = screen_normalization_adjustments(annual_financials, normalization_adjustments)
    analytics = calculate_quality_of_earnings(
        annual_financials,
        calc["multi_year_history"],
        adjustments=screened["accepted"],
        prior_total_assets=prior_balance_sheet.total_assets if prior_balance_sheet else None,
    )
    findings = detect_forensic_findings(annual_financials, calc["multi_year_history"], analytics, balance_sheet)

    result = FinancialAuditResult(
        **calc, **analytics, forensic_findings=findings,
        rejected_normalization_adjustments=screened["rejected"],
    )
    logger.info(
        f"audit_financial_metrics: FY{balance_sheet.fiscal_year} net debt ${result.balance_sheet.net_debt:,.2f}M | "
        f"ROIC {result.profitability_and_return_ratios.roic_pct}% | findings {len(findings)}"
    )
    return result.model_dump()


@tool
def audit_financial_metrics_tool(
    annual_financials: List[AnnualFinancialInput],
    balance_sheet: BalanceSheetInput,
    prior_balance_sheet: Optional[PriorBalanceSheetInput] = None,
    normalization_adjustments: Optional[List[NormalizationAdjustment]] = None,
) -> Dict[str, Any]:
    """
    Audit 10-K financial statements: deterministic ratios, quality-of-earnings analytics and forensic findings.

    All dollar amounts in $ MILLIONS, shares in MILLIONS, CapEx as a POSITIVE magnitude.

    Args:
        annual_financials: Contiguous fiscal years. Required per year: fiscal_year, revenue, operating_income,
            net_income (attributable to common), operating_cash_flow, capital_expenditures. Supply every other
            line item that exists in the statements: gross_profit or cost_of_revenue, operating_expenses,
            pretax_income, income_tax_expense, net_income_total, depreciation_amortization,
            stock_based_compensation, deferred_income_taxes, accounts_receivable, inventories, accounts_payable,
            total_assets, diluted_weighted_shares, period_end_shares_outstanding.
        balance_sheet: Latest-year snapshot: cash_and_equivalents, marketable_securities (ALL short-term
            investments + non-current marketable securities), short_term_debt, long_term_debt,
            stockholders_equity, weighted_diluted_shares and/or period_end_shares_outstanding, current_assets,
            current_liabilities, total_assets.
        prior_balance_sheet: Prior-year snapshot (enables average-capital ROIC/ROE and accruals).
        normalization_adjustments: Non-recurring / non-core items: positive `amount` plus `direction`
            ("inflated_reported_earnings" for gains/credits/tax benefits, "depressed_reported_earnings"
            for impairments/restructuring/litigation charges).
    """
    return audit_financial_metrics(
        annual_financials=annual_financials,
        balance_sheet=balance_sheet,
        prior_balance_sheet=prior_balance_sheet,
        normalization_adjustments=normalization_adjustments,
    )


def run_audit_from_inputs(
    inputs: Dict[str, Any],
    normalization_adjustments: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Validates plain-dict audit inputs into Pydantic models and runs `audit_financial_metrics`."""
    annual_financials, balance_sheet, prior_balance_sheet = validate_audit_inputs(inputs)
    return audit_financial_metrics(
        annual_financials=annual_financials,
        balance_sheet=balance_sheet,
        prior_balance_sheet=prior_balance_sheet,
        normalization_adjustments=[
            a if isinstance(a, NormalizationAdjustment) else NormalizationAdjustment.model_validate(a)
            for a in (normalization_adjustments or [])
        ],
    )
