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
    growth_pct,
    pct_of,
)

logger = logging.getLogger("finance_agent.tools.financial_math")


# Forensic thresholds (institutional quality-of-earnings screens)
CAPEX_DRIVEN_FCF_THRESHOLD_PCT = 75.0
DILUTED_SHARE_GROWTH_THRESHOLD_PCT = 3.0
PERIOD_END_SHARE_GROWTH_THRESHOLD_PCT = 5.0



# ==============================================================================
# 2. Output Schemas
# ==============================================================================
class BalanceSheetResult(BaseModel):
    fiscal_year: int
    cash_and_equivalents: float
    marketable_securities: float
    total_liquid_cash: float
    short_term_debt: float
    long_term_debt: float
    total_debt: float
    prior_total_debt: Optional[float] = Field(None, description="Prior-year gross debt (for average-debt Kd)")
    net_debt: float
    net_cash_position: bool
    stockholders_equity: float
    noncontrolling_interests: Optional[float] = None
    valuation_shares_outstanding: float = Field(
        ..., description="Share count used for per-share valuation (millions)"
    )
    share_count_source: str
    weighted_diluted_shares: Optional[float] = None
    period_end_shares_outstanding: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_assets: Optional[float] = None


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


class SolvencyRatiosResult(BaseModel):
    ebitda: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    net_debt_to_ebitda_interpretation: str
    current_ratio: Optional[float] = None
    total_debt_to_equity: Optional[float] = None


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


class YearCapitalIntensity(BaseModel):
    fiscal_year: int
    capex_to_revenue_pct: Optional[float] = None
    capex_to_depreciation: Optional[float] = None
    ocf_change: Optional[float] = None
    capex_change: Optional[float] = None
    fcf_change: Optional[float] = None
    fcf_change_from_capex_pct: Optional[float] = Field(
        None, description="Share of the FCF change explained by the change in capex"
    )


class YearShareCount(BaseModel):
    fiscal_year: int
    diluted_weighted_shares: Optional[float] = None
    diluted_share_growth_pct: Optional[float] = None
    period_end_shares_outstanding: Optional[float] = None
    period_end_share_growth_pct: Optional[float] = None


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

    # Effective tax rate with provenance
    latest_tax = latest_tax_rate(latest_input)
    tax_rate, tax_source = latest_tax.rate, latest_tax.source

    # Liquidity bridge
    sec = balance_sheet.marketable_securities or 0.0
    st_debt = balance_sheet.short_term_debt or 0.0
    lt_debt = balance_sheet.long_term_debt or 0.0
    total_liquid = round(balance_sheet.cash_and_equivalents + sec, 2)
    total_debt = round(st_debt + lt_debt, 2)
    net_debt = round(total_debt - total_liquid, 2)
    net_cash_pos = bool(net_debt < 0)

    # Valuation share count: never below the shares actually outstanding at period end.
    wd = balance_sheet.weighted_diluted_shares
    pe = balance_sheet.period_end_shares_outstanding
    if wd is not None and pe is not None:
        if pe > wd:
            valuation_shares, share_source = pe, "period_end_basic_exceeds_weighted_diluted"
        else:
            valuation_shares, share_source = wd, "weighted_average_diluted"
    elif wd is not None:
        valuation_shares, share_source = wd, "weighted_average_diluted"
    else:
        valuation_shares, share_source = pe, "period_end_basic_only"

    # Returns on (average) capital
    nopat = round(latest.operating_income * (1.0 - tax_rate), 2)
    equity = balance_sheet.stockholders_equity
    invested_capital = round(total_debt + equity - total_liquid, 2)
    gross_capital = total_debt + equity
    avg_equity = equity
    capital_basis: Literal["average", "ending"] = "ending"
    if prior_balance_sheet is not None:
        prior_liquid = prior_balance_sheet.cash_and_equivalents + (prior_balance_sheet.marketable_securities or 0.0)
        prior_debt = (prior_balance_sheet.short_term_debt or 0.0) + (prior_balance_sheet.long_term_debt or 0.0)
        prior_ic = prior_debt + prior_balance_sheet.stockholders_equity - prior_liquid
        if prior_ic > 0 and invested_capital > 0 and prior_balance_sheet.stockholders_equity > 0 and equity > 0:
            invested_capital = round((invested_capital + prior_ic) / 2.0, 2)
            gross_capital = (gross_capital + prior_debt + prior_balance_sheet.stockholders_equity) / 2.0
            avg_equity = (equity + prior_balance_sheet.stockholders_equity) / 2.0
            capital_basis = "average"

    equity_status = "positive_book_equity"
    roic_pct = roic_gross_pct = roe_pct = total_debt_equity = None
    if equity < 0:
        equity_status = "negative_book_equity_ratios_suppressed"
        logger.warning(
            f"calculate_financial_ratios: Stockholders' equity is negative (${equity:,.1f}M). "
            f"Suppressing ROIC, ROE, and Debt/Equity."
        )
    elif equity == 0:
        equity_status = "zero_book_equity_ratios_suppressed"
    else:
        if invested_capital > 0:
            roic_pct = round(nopat / invested_capital * 100.0, 2)
        if gross_capital > 0:
            roic_gross_pct = round(nopat / gross_capital * 100.0, 2)
        roe_pct = round(latest.net_income / avg_equity * 100.0, 2)
        total_debt_equity = round(total_debt / equity, 2)

    ebitda = net_debt_ebitda = None
    ebitda_interp = "ebitda_not_available"
    if latest_input.depreciation_amortization is not None:
        ebitda = round(latest.operating_income + latest_input.depreciation_amortization, 2)
        if ebitda > 0:
            net_debt_ebitda = round(net_debt / ebitda, 2)
            ebitda_interp = "net_cash_surplus" if net_cash_pos else "leverage_ratio"
        elif ebitda < 0:
            ebitda_interp = "negative_ebitda"

    current_ratio = None
    if balance_sheet.current_assets is not None and balance_sheet.current_liabilities:
        current_ratio = round(balance_sheet.current_assets / balance_sheet.current_liabilities, 2)

    return {
        "multi_year_history": history,
        "balance_sheet": BalanceSheetResult(
            fiscal_year=balance_sheet.fiscal_year,
            cash_and_equivalents=balance_sheet.cash_and_equivalents,
            marketable_securities=sec,
            total_liquid_cash=total_liquid,
            short_term_debt=st_debt,
            long_term_debt=lt_debt,
            total_debt=total_debt,
            prior_total_debt=(
                round((prior_balance_sheet.short_term_debt or 0.0) + (prior_balance_sheet.long_term_debt or 0.0), 2)
                if prior_balance_sheet is not None else None
            ),
            net_debt=net_debt,
            net_cash_position=net_cash_pos,
            stockholders_equity=equity,
            noncontrolling_interests=balance_sheet.noncontrolling_interests,
            valuation_shares_outstanding=valuation_shares,
            share_count_source=share_source,
            weighted_diluted_shares=wd,
            period_end_shares_outstanding=pe,
            current_assets=balance_sheet.current_assets,
            current_liabilities=balance_sheet.current_liabilities,
            total_assets=balance_sheet.total_assets,
        ),
        "profitability_and_return_ratios": ProfitabilityRatiosResult(
            effective_tax_rate_pct=round(tax_rate * 100.0, 2),
            tax_rate_source=tax_source,
            equity_status=equity_status,
            nopat=nopat,
            invested_capital=invested_capital,
            capital_basis=capital_basis,
            roic_pct=roic_pct,
            roic_on_gross_capital_pct=roic_gross_pct,
            roe_pct=roe_pct,
            latest_gross_margin_pct=latest.gross_margin_pct,
            latest_operating_margin_pct=latest.operating_margin_pct,
            latest_net_margin_pct=latest.net_margin_pct,
        ),
        "solvency_and_liquidity_ratios": SolvencyRatiosResult(
            ebitda=ebitda,
            net_debt_to_ebitda=net_debt_ebitda,
            net_debt_to_ebitda_interpretation=ebitda_interp,
            current_ratio=current_ratio,
            total_debt_to_equity=total_debt_equity,
        ),
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
    hist_map = {h.fiscal_year: h for h in history}
    normalized_by_year = {r.fiscal_year: r for r in normalize_earnings(annual_financials, history, adjustments)}
    cash_by_year = {r.fiscal_year: r for r in measure_cash_conversion(annual_financials, history, prior_total_assets)}
    tax_by_year = {r.fiscal_year: r for r in measure_tax_quality(annual_financials)}
    eq: List[YearEarningsQuality] = []
    ci: List[YearCapitalIntensity] = []
    sh: List[YearShareCount] = []

    prev: Optional[AnnualFinancialInput] = None
    for f in inputs:
        h = hist_map[f.fiscal_year]
        rev = f.revenue

        # --- Earnings quality
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

        # --- Capital intensity & FCF bridge
        ocf_chg = capex_chg = fcf_chg = from_capex = None
        if prev is not None:
            prev_h = hist_map[prev.fiscal_year]
            ocf_chg = round(f.operating_cash_flow - prev.operating_cash_flow, 2)
            capex_chg = round(f.capital_expenditures - prev.capital_expenditures, 2)
            fcf_chg = round(h.free_cash_flow - prev_h.free_cash_flow, 2)
            from_capex = pct_of(-capex_chg, fcf_chg) if fcf_chg else None
        ci.append(YearCapitalIntensity(
            fiscal_year=f.fiscal_year,
            capex_to_revenue_pct=pct_of(f.capital_expenditures, rev),
            capex_to_depreciation=round(f.capital_expenditures / f.depreciation_amortization, 2)
            if f.depreciation_amortization else None,
            ocf_change=ocf_chg,
            capex_change=capex_chg,
            fcf_change=fcf_chg,
            fcf_change_from_capex_pct=from_capex,
        ))

        # --- Share count drift
        sh.append(YearShareCount(
            fiscal_year=f.fiscal_year,
            diluted_weighted_shares=f.diluted_weighted_shares,
            diluted_share_growth_pct=growth_pct(f.diluted_weighted_shares, prev.diluted_weighted_shares) if prev else None,
            period_end_shares_outstanding=f.period_end_shares_outstanding,
            period_end_share_growth_pct=growth_pct(f.period_end_shares_outstanding, prev.period_end_shares_outstanding)
            if prev else None,
        ))

        prev = f

    return {
        "earnings_quality": eq,
        "working_capital": measure_working_capital(annual_financials),
        "capital_intensity": ci,
        "cost_structure": measure_cost_structure(annual_financials, history),
        "share_count_history": sh,
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
    ci = {r.fiscal_year: r for r in analytics["capital_intensity"]}
    sh = {r.fiscal_year: r for r in analytics["share_count_history"]}

    def add(code, severity, year, metric, value, threshold, message):
        findings.append(ForensicFinding(
            code=code, severity=severity, fiscal_year=year, metric=metric,
            value=round(value, 2) if value is not None else None, threshold=threshold, message=message,
        ))

    if balance_sheet is not None and balance_sheet.stockholders_equity < 0:
        add("NEGATIVE_BOOK_EQUITY", "high", balance_sheet.fiscal_year, "stockholders_equity",
            balance_sheet.stockholders_equity, 0.0,
            f"NEGATIVE BOOK EQUITY WARNING: Stockholders' equity is negative "
            f"(${balance_sheet.stockholders_equity:,.1f}M). ROE, ROIC and Debt-to-Equity are suppressed as "
            f"uninterpretable.")

    for yr, c in ci.items():
        if c.fcf_change is not None and c.fcf_change > 0 and (c.ocf_change or 0.0) <= 0 \
                and (c.capex_change or 0.0) < 0 and (c.fcf_change_from_capex_pct or 0.0) > CAPEX_DRIVEN_FCF_THRESHOLD_PCT:
            add("CAPEX_DRIVEN_FCF", "medium", yr, "fcf_change_from_capex_pct", c.fcf_change_from_capex_pct,
                CAPEX_DRIVEN_FCF_THRESHOLD_PCT,
                f"CAPEX-DRIVEN FCF (FY{yr}): Free cash flow rose ${c.fcf_change:,.1f}M while operating cash flow "
                f"changed ${c.ocf_change:,.1f}M; a ${abs(c.capex_change):,.1f}M capex reduction explains "
                f"{c.fcf_change_from_capex_pct:.1f}% of the improvement, so it reflects lower reinvestment rather "
                f"than stronger cash generation.")

    capex_da = [ci[h.fiscal_year].capex_to_depreciation for h in history[-2:]]
    if len(capex_da) == 2 and all(v is not None and v < 1.0 for v in capex_da):
        add("UNDERINVESTMENT", "info", history[-1].fiscal_year, "capex_to_depreciation", capex_da[-1], 1.0,
            f"UNDERINVESTMENT SIGNAL: Capex has run below depreciation for two consecutive years "
            f"(latest {capex_da[-1]:.2f}x), implying a shrinking asset base.")

    for yr, s in sh.items():
        if s.diluted_share_growth_pct is not None and s.diluted_share_growth_pct > DILUTED_SHARE_GROWTH_THRESHOLD_PCT:
            add("SHARE_DILUTION", "medium", yr, "diluted_share_growth_pct", s.diluted_share_growth_pct,
                DILUTED_SHARE_GROWTH_THRESHOLD_PCT,
                f"SHARE DILUTION (FY{yr}): Weighted diluted shares grew {s.diluted_share_growth_pct:.2f}% "
                f"to {s.diluted_weighted_shares:,.1f}M.")
        if s.period_end_share_growth_pct is not None and \
                s.period_end_share_growth_pct > PERIOD_END_SHARE_GROWTH_THRESHOLD_PCT:
            add("SHARE_DILUTION", "high", yr, "period_end_share_growth_pct", s.period_end_share_growth_pct,
                PERIOD_END_SHARE_GROWTH_THRESHOLD_PCT,
                f"SHARE COUNT EXPANSION (FY{yr}): Shares outstanding at year end rose "
                f"{s.period_end_share_growth_pct:.2f}% to {s.period_end_shares_outstanding:,.1f}M; investigate "
                f"equity awards, conversions or issuances, and use the period-end count for per-share value.")

    findings.extend(screen_cash_conversion(history, analytics["earnings_quality"]))
    findings.extend(screen_tax_quality(annual_inputs, analytics["earnings_quality"]))
    findings.extend(screen_earnings_normalization(analytics["earnings_quality"]))
    findings.extend(screen_margins_and_costs(history, analytics["cost_structure"]))
    findings.extend(screen_working_capital(annual_inputs, history, analytics["working_capital"]))
    findings.extend(screen_revenue_quality(annual_inputs, history))
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
