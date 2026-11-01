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
import math
import re
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
from app.agents.specialized.financial_auditor.findings import ForensicFinding, Severity, rank_findings

logger = logging.getLogger("finance_agent.tools.financial_math")

STATUTORY_TAX_RATE = 0.21

# Forensic thresholds (institutional quality-of-earnings screens)
NON_OPERATING_TO_EBIT_THRESHOLD_PCT = 20.0
NORMALIZATION_TO_EBIT_THRESHOLD_PCT = 20.0
ETR_LOW_PCT = 5.0
ETR_HIGH_PCT = 40.0
DEFERRED_TAX_TO_NI_THRESHOLD_PCT = 25.0
CAPEX_DRIVEN_FCF_THRESHOLD_PCT = 75.0
SBC_TO_OCF_THRESHOLD_PCT = 15.0
SBC_TO_EBIT_THRESHOLD_PCT = 40.0
DILUTED_SHARE_GROWTH_THRESHOLD_PCT = 3.0
PERIOD_END_SHARE_GROWTH_THRESHOLD_PCT = 5.0
OPEX_VS_REVENUE_GROWTH_THRESHOLD_PP = 10.0
MARGIN_SINGLE_YEAR_DROP_BPS = 300.0
MARGIN_CUMULATIVE_DROP_BPS = 300.0
SLOAN_ACCRUALS_THRESHOLD_PCT = 10.0
WORKING_CAPITAL_DAYS_THRESHOLD_PCT = 15.0
FCF_CONVERSION_WEAK_PCT = 70.0



class NormalizationAdjustment(BaseModel):
    """
    A non-recurring or non-core item to strip from reported earnings.

    - `amount`: positive magnitude in $ millions, exactly as disclosed.
    - `direction`: "inflated_reported_earnings" for gains, credit sales, tax benefits and
      valuation-allowance releases (normalized earnings are LOWER); "depressed_reported_earnings"
      for charges such as impairments, restructuring and litigation (normalized earnings are HIGHER).
    """

    fiscal_year: int
    label: str
    amount: float = Field(..., gt=0, description="Positive magnitude in $ millions")
    direction: Literal["inflated_reported_earnings", "depressed_reported_earnings"]
    affects: Literal["operating_income", "net_income"] = "operating_income"
    rationale: Optional[str] = None
    source_chunk_id: Optional[str] = None

    @property
    def signed_effect(self) -> float:
        """Effect on reported earnings (+ inflated, - depressed); normalized = reported - signed_effect."""
        return self.amount if self.direction == "inflated_reported_earnings" else -self.amount


# ==============================================================================
# 2. Output Schemas
# ==============================================================================
class YearFinancialsResult(BaseModel):
    fiscal_year: int
    revenue: float
    gross_profit: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    operating_income: float
    operating_margin_pct: float
    net_income: float
    net_margin_pct: float
    operating_cash_flow: float
    capital_expenditures: float
    free_cash_flow: float
    depreciation_amortization: Optional[float] = None
    interest_expense: Optional[float] = None
    fcf_conversion_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    operating_income_growth_pct: Optional[float] = None
    fcf_growth_pct: Optional[float] = None


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


class YearWorkingCapital(BaseModel):
    fiscal_year: int
    dso_days: Optional[float] = None
    dio_days: Optional[float] = None
    dpo_days: Optional[float] = None
    cash_conversion_cycle_days: Optional[float] = None
    dso_change_pct: Optional[float] = None
    dio_change_pct: Optional[float] = None


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


class YearCostStructure(BaseModel):
    fiscal_year: int
    operating_expenses: Optional[float] = None
    operating_expense_growth_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    opex_growth_minus_revenue_growth_pp: Optional[float] = None
    gross_margin_change_bps: Optional[float] = None
    operating_margin_change_bps: Optional[float] = None


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
def _validate_finite_number(val: Any, name: str) -> float:
    if val is None:
        raise ValueError(f"{name} cannot be None.")
    if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        raise ValueError(f"{name} must be a valid finite number, got: {val}")
    return float(val)


def _pct(num: Optional[float], den: Optional[float], digits: int = 2) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return round(num / den * 100.0, digits)


def _growth(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    if cur is None or prev is None or prev == 0:
        return None
    return round((cur - prev) / abs(prev) * 100.0, 2)


def _year_tax_rate(inp: AnnualFinancialInput) -> Optional[float]:
    """Derived effective tax rate (decimal) when pre-tax income is positive and tax is an expense."""
    if inp.pretax_income is None or inp.income_tax_expense is None:
        return None
    if inp.pretax_income <= 0 or inp.income_tax_expense < 0:
        return None
    return inp.income_tax_expense / inp.pretax_income


def _cost_of_revenue(inp: AnnualFinancialInput) -> Optional[float]:
    if inp.cost_of_revenue is not None:
        return inp.cost_of_revenue
    if inp.gross_profit is not None:
        return inp.revenue - inp.gross_profit
    return None


def _gross_profit(inp: AnnualFinancialInput) -> Optional[float]:
    if inp.gross_profit is not None:
        return inp.gross_profit
    if inp.cost_of_revenue is not None:
        return round(inp.revenue - inp.cost_of_revenue, 4)
    return None


def _operating_expenses(inp: AnnualFinancialInput) -> Optional[float]:
    if inp.operating_expenses is not None:
        return inp.operating_expenses
    gp = _gross_profit(inp)
    return round(gp - inp.operating_income, 4) if gp is not None else None


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
    history: List[YearFinancialsResult] = []
    prev: Optional[AnnualFinancialInput] = None
    prev_fcf: Optional[float] = None

    for f in sorted_years:
        rev = _validate_finite_number(f.revenue, f"revenue ({f.fiscal_year})")
        ebit = _validate_finite_number(f.operating_income, f"operating_income ({f.fiscal_year})")
        ni = _validate_finite_number(f.net_income, f"net_income ({f.fiscal_year})")
        ocf = _validate_finite_number(f.operating_cash_flow, f"operating_cash_flow ({f.fiscal_year})")
        capex = _validate_finite_number(f.capital_expenditures, f"capital_expenditures ({f.fiscal_year})")
        gp = _gross_profit(f)
        fcf = round(ocf - capex, 2)

        history.append(YearFinancialsResult(
            fiscal_year=f.fiscal_year,
            revenue=rev,
            gross_profit=gp,
            gross_margin_pct=_pct(gp, rev),
            operating_income=ebit,
            operating_margin_pct=_pct(ebit, rev) or 0.0,
            net_income=ni,
            net_margin_pct=_pct(ni, rev) or 0.0,
            operating_cash_flow=ocf,
            capital_expenditures=capex,
            free_cash_flow=fcf,
            depreciation_amortization=f.depreciation_amortization,
            interest_expense=abs(f.interest_expense) if f.interest_expense is not None else None,
            fcf_conversion_pct=_pct(fcf, ni),
            revenue_growth_pct=_growth(rev, prev.revenue) if prev else None,
            operating_income_growth_pct=_growth(ebit, prev.operating_income) if prev else None,
            fcf_growth_pct=_growth(fcf, prev_fcf) if prev else None,
        ))
        prev, prev_fcf = f, fcf

    latest_input = sorted_years[-1]
    latest = history[-1]

    # Effective tax rate with provenance
    tax_rate, tax_source = STATUTORY_TAX_RATE, "statutory_default_not_provided"
    if latest_input.pretax_income is not None and latest_input.income_tax_expense is not None:
        if latest_input.pretax_income <= 0:
            tax_source = "statutory_fallback_due_to_pretax_loss"
        elif latest_input.income_tax_expense < 0:
            tax_source = "statutory_fallback_due_to_tax_benefit"
        else:
            tax_rate = round(latest_input.income_tax_expense / latest_input.pretax_income, 4)
            tax_source = "derived_from_10k"

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
_AGGREGATE_LINE_LABEL = re.compile(
    r"non[- ]?operating|non[- ]?core income|other income|other \(expense\)|interest (and other )?income|"
    r"total other|income dependence"
)
_RECURRING_COST_LABEL = re.compile(
    r"research and development|selling, general|general and administrative|stock[- ]based|share[- ]based"
)
_TAX_LABEL = re.compile(r"\btax(?:es)?\b")
_DISCRETE_TAX_EVENT = re.compile(
    r"valuation[- ]allowance|remeasure|re-measure|settlement|tax (?:act|reform|law)|tax cuts|tcja|enact|"
    r"repatriat|transition tax|one[- ]time|discrete|intra[- ]entity|ip transfer|uncertain tax|statute"
)


def screen_normalization_adjustments(
    annual_financials: List[AnnualFinancialInput],
    adjustments: Optional[List[NormalizationAdjustment]],
) -> Dict[str, List[Any]]:
    """
    Deterministically rejects normalization items that are not discrete non-recurring disclosures:
    out-of-window years, aggregate non-operating subtotals (already measured by the
    NON_CORE_INCOME_DEPENDENCE screen), recurring operating costs and the recurring income tax
    provision (only discrete tax events such as valuation-allowance releases qualify).
    """
    inputs = {f.fiscal_year: f for f in annual_financials}
    accepted: List[NormalizationAdjustment] = []
    rejected: List[Dict[str, Any]] = []
    for adj in adjustments or []:
        reason = None
        f = inputs.get(adj.fiscal_year)
        label = adj.label.lower()
        tax_provision = abs(f.income_tax_expense) if f is not None and f.income_tax_expense else None
        if f is None:
            reason = f"FY{adj.fiscal_year} is outside the audited window {sorted(inputs)}"
        elif _AGGREGATE_LINE_LABEL.search(label):
            reason = "aggregate non-operating line; measured deterministically by NON_CORE_INCOME_DEPENDENCE"
        elif _RECURRING_COST_LABEL.search(label):
            reason = "recurring operating cost; not a non-recurring item"
        elif _TAX_LABEL.search(label) and not _DISCRETE_TAX_EVENT.search(f"{label} {(adj.rationale or '').lower()}"):
            reason = (
                "recurring income tax provision; only discrete tax events (valuation-allowance release, "
                "remeasurement, settlement, tax-law change) qualify, and the effective tax rate is measured separately"
            )
        elif tax_provision and abs(adj.amount - tax_provision) <= 0.005 * tax_provision:
            reason = "amount equals the full income tax provision, a recurring expense"
        elif f.pretax_income is not None:
            non_op = f.pretax_income - f.operating_income
            if non_op and abs(adj.amount - abs(non_op)) <= 0.005 * abs(non_op):
                reason = "amount equals the aggregate non-operating income subtotal"
        if reason:
            rejected.append({**adj.model_dump(), "rejection": reason})
        else:
            accepted.append(adj)
    return {"accepted": accepted, "rejected": rejected}


def calculate_quality_of_earnings(
    annual_financials: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
    adjustments: Optional[List[NormalizationAdjustment]] = None,
    prior_total_assets: Optional[float] = None,
) -> Dict[str, List[BaseModel]]:
    """Earnings quality, working capital, capital intensity, cost structure and share-count analytics."""
    inputs = sorted(annual_financials, key=lambda x: x.fiscal_year)
    hist_map = {h.fiscal_year: h for h in history}
    adj_by_year: Dict[int, List[NormalizationAdjustment]] = {}
    for adj in adjustments or []:
        if adj.fiscal_year not in hist_map:
            raise ValueError(
                f"normalization adjustment '{adj.label}' targets FY{adj.fiscal_year}, "
                f"outside the audited window {sorted(hist_map)}."
            )
        adj_by_year.setdefault(adj.fiscal_year, []).append(adj)

    eq: List[YearEarningsQuality] = []
    wc: List[YearWorkingCapital] = []
    ci: List[YearCapitalIntensity] = []
    cs: List[YearCostStructure] = []
    sh: List[YearShareCount] = []

    prev: Optional[AnnualFinancialInput] = None
    prev_wc: Optional[YearWorkingCapital] = None
    for f in inputs:
        h = hist_map[f.fiscal_year]
        rev = f.revenue

        # --- Earnings quality
        ni_for_accruals = f.net_income_total if f.net_income_total is not None else f.net_income
        accruals = round(ni_for_accruals - f.operating_cash_flow, 2)
        ta_prev = prev.total_assets if prev is not None else prior_total_assets
        avg_ta = None
        if f.total_assets is not None:
            avg_ta = (f.total_assets + ta_prev) / 2.0 if ta_prev is not None else f.total_assets
        sbc = f.stock_based_compensation
        non_op = round(f.pretax_income - f.operating_income, 2) if f.pretax_income is not None else None
        year_rate = _year_tax_rate(f)
        etr_pct = _pct(f.income_tax_expense, f.pretax_income) if (
            f.pretax_income not in (None, 0) and f.income_tax_expense is not None
        ) else None

        year_adj = adj_by_year.get(f.fiscal_year, [])
        ebit_adj = round(sum(a.signed_effect for a in year_adj if a.affects == "operating_income"), 2)
        ni_direct_adj = round(sum(a.signed_effect for a in year_adj if a.affects == "net_income"), 2)
        after_tax_factor = 1.0 - (year_rate if year_rate is not None else STATUTORY_TAX_RATE)
        ni_total_adj = round(ebit_adj * after_tax_factor + ni_direct_adj, 2)
        norm_ebit = round(f.operating_income - ebit_adj, 2)
        norm_ni = round(f.net_income - ni_total_adj, 2)
        abs_ebit_adj = sum(abs(a.amount) for a in year_adj if a.affects == "operating_income")

        eq.append(YearEarningsQuality(
            fiscal_year=f.fiscal_year,
            accruals=accruals,
            sloan_accruals_ratio_pct=_pct(accruals, avg_ta),
            stock_based_compensation=sbc,
            sbc_to_ocf_pct=_pct(sbc, f.operating_cash_flow) if f.operating_cash_flow > 0 else None,
            sbc_to_operating_income_pct=_pct(sbc, f.operating_income) if f.operating_income > 0 else None,
            sbc_adjusted_fcf=round(h.free_cash_flow - sbc, 2) if sbc is not None else None,
            non_operating_income=non_op,
            non_operating_to_operating_income_pct=_pct(non_op, f.operating_income) if f.operating_income > 0 else None,
            effective_tax_rate_pct=etr_pct,
            deferred_tax_to_net_income_pct=_pct(f.deferred_income_taxes, abs(f.net_income)) if f.net_income else None,
            adjustments=year_adj,
            operating_income_adjustments_total=ebit_adj,
            net_income_adjustments_total=ni_total_adj,
            adjustments_pct_of_operating_income=_pct(abs_ebit_adj, abs(f.operating_income)) if year_adj else None,
            normalized_operating_income=norm_ebit,
            normalized_operating_margin_pct=_pct(norm_ebit, rev) or 0.0,
            normalized_net_income=norm_ni,
            normalized_net_margin_pct=_pct(norm_ni, rev) or 0.0,
        ))

        # --- Working capital days (ending-balance basis)
        cogs = _cost_of_revenue(f)
        dso = round(f.accounts_receivable / rev * 365.0, 1) if f.accounts_receivable is not None and rev > 0 else None
        dio = round(f.inventories / cogs * 365.0, 1) if f.inventories is not None and cogs and cogs > 0 else None
        dpo = round(f.accounts_payable / cogs * 365.0, 1) if f.accounts_payable is not None and cogs and cogs > 0 else None
        ccc = round((dso or 0.0) + (dio or 0.0) - (dpo or 0.0), 1) if None not in (dso, dpo) else None
        year_wc = YearWorkingCapital(
            fiscal_year=f.fiscal_year,
            dso_days=dso,
            dio_days=dio,
            dpo_days=dpo,
            cash_conversion_cycle_days=ccc,
            dso_change_pct=_growth(dso, prev_wc.dso_days) if prev_wc else None,
            dio_change_pct=_growth(dio, prev_wc.dio_days) if prev_wc else None,
        )
        wc.append(year_wc)

        # --- Capital intensity & FCF bridge
        ocf_chg = capex_chg = fcf_chg = from_capex = None
        if prev is not None:
            prev_h = hist_map[prev.fiscal_year]
            ocf_chg = round(f.operating_cash_flow - prev.operating_cash_flow, 2)
            capex_chg = round(f.capital_expenditures - prev.capital_expenditures, 2)
            fcf_chg = round(h.free_cash_flow - prev_h.free_cash_flow, 2)
            from_capex = _pct(-capex_chg, fcf_chg) if fcf_chg else None
        ci.append(YearCapitalIntensity(
            fiscal_year=f.fiscal_year,
            capex_to_revenue_pct=_pct(f.capital_expenditures, rev),
            capex_to_depreciation=round(f.capital_expenditures / f.depreciation_amortization, 2)
            if f.depreciation_amortization else None,
            ocf_change=ocf_chg,
            capex_change=capex_chg,
            fcf_change=fcf_chg,
            fcf_change_from_capex_pct=from_capex,
        ))

        # --- Cost structure
        opex = _operating_expenses(f)
        opex_growth = rev_growth = gm_bps = om_bps = None
        if prev is not None:
            prev_h = hist_map[prev.fiscal_year]
            opex_growth = _growth(opex, _operating_expenses(prev))
            rev_growth = h.revenue_growth_pct
            if h.gross_margin_pct is not None and prev_h.gross_margin_pct is not None:
                gm_bps = round((h.gross_margin_pct - prev_h.gross_margin_pct) * 100.0, 0)
            om_bps = round((h.operating_margin_pct - prev_h.operating_margin_pct) * 100.0, 0)
        cs.append(YearCostStructure(
            fiscal_year=f.fiscal_year,
            operating_expenses=opex,
            operating_expense_growth_pct=opex_growth,
            revenue_growth_pct=rev_growth,
            opex_growth_minus_revenue_growth_pp=round(opex_growth - rev_growth, 2)
            if opex_growth is not None and rev_growth is not None else None,
            gross_margin_change_bps=gm_bps,
            operating_margin_change_bps=om_bps,
        ))

        # --- Share count drift
        sh.append(YearShareCount(
            fiscal_year=f.fiscal_year,
            diluted_weighted_shares=f.diluted_weighted_shares,
            diluted_share_growth_pct=_growth(f.diluted_weighted_shares, prev.diluted_weighted_shares) if prev else None,
            period_end_shares_outstanding=f.period_end_shares_outstanding,
            period_end_share_growth_pct=_growth(f.period_end_shares_outstanding, prev.period_end_shares_outstanding)
            if prev else None,
        ))

        prev, prev_wc = f, year_wc

    return {
        "earnings_quality": eq,
        "working_capital": wc,
        "capital_intensity": ci,
        "cost_structure": cs,
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
    inputs = {f.fiscal_year: f for f in annual_inputs}
    eq = {r.fiscal_year: r for r in analytics["earnings_quality"]}
    wc = {r.fiscal_year: r for r in analytics["working_capital"]}
    ci = {r.fiscal_year: r for r in analytics["capital_intensity"]}
    cs = {r.fiscal_year: r for r in analytics["cost_structure"]}
    sh = {r.fiscal_year: r for r in analytics["share_count_history"]}

    def add(code, severity, year, metric, value, threshold, message):
        findings.append(ForensicFinding(
            code=code, severity=severity, fiscal_year=year, metric=metric,
            value=round(value, 2) if value is not None else None, threshold=threshold, message=message,
        ))

    # Net income up while OCF down
    for prev, cur in zip(history, history[1:]):
        ni_g = _growth(cur.net_income, prev.net_income)
        ocf_g = _growth(cur.operating_cash_flow, prev.operating_cash_flow)
        if ni_g is not None and ocf_g is not None and ni_g > 0.0 and ocf_g < 0.0:
            add("EARNINGS_QUALITY_DIVERGENCE", "high", cur.fiscal_year, "net_income_vs_ocf_growth_pct", ni_g, 0.0,
                f"EARNINGS QUALITY DIVERGENCE (FY{cur.fiscal_year}): Net income increased by +{ni_g:.2f}% "
                f"(${prev.net_income:,.1f}M -> ${cur.net_income:,.1f}M) while operating cash flow declined by "
                f"{ocf_g:.2f}% (${prev.operating_cash_flow:,.1f}M -> ${cur.operating_cash_flow:,.1f}M), indicating "
                f"accrual expansion, non-operating gains, or tax/working-capital cash timing outflows.")

    profitable = [h for h in history if h.net_income > 0 and h.fcf_conversion_pct is not None]
    if len(profitable) >= 2 and all(h.fcf_conversion_pct < FCF_CONVERSION_WEAK_PCT for h in profitable):
        avg_conv = sum(h.fcf_conversion_pct for h in profitable) / len(profitable)
        add("WEAK_CASH_CONVERSION", "medium", history[-1].fiscal_year, "avg_fcf_conversion_pct", avg_conv,
            FCF_CONVERSION_WEAK_PCT,
            f"WEAK CASH CONVERSION WARNING: Free cash flow conversion (FCF / Net Income) has been chronically "
            f"weak (< 70%) across all reported years with positive earnings (averaging {avg_conv:.1f}%), "
            f"signaling high capital intensity or poor earnings-to-cash realization.")

    if len(history) >= 2 and all(h.operating_cash_flow < h.net_income for h in history):
        add("CHRONIC_ACCRUAL_DEFICIT", "high", history[-1].fiscal_year, "ocf_below_net_income_years",
            float(len(history)), None,
            "CHRONIC ACCRUAL DEFICIT: Operating cash flow was strictly lower than net income in every "
            "reported fiscal year, signaling persistently low cash conversion quality.")

    if balance_sheet is not None and balance_sheet.stockholders_equity < 0:
        add("NEGATIVE_BOOK_EQUITY", "high", balance_sheet.fiscal_year, "stockholders_equity",
            balance_sheet.stockholders_equity, 0.0,
            f"NEGATIVE BOOK EQUITY WARNING: Stockholders' equity is negative "
            f"(${balance_sheet.stockholders_equity:,.1f}M). ROE, ROIC and Debt-to-Equity are suppressed as "
            f"uninterpretable.")

    for prev, cur in zip(history, history[1:]):
        p_in, c_in = inputs[prev.fiscal_year], inputs[cur.fiscal_year]
        rev_g = cur.revenue_growth_pct or 0.0
        if p_in.accounts_receivable and c_in.accounts_receivable is not None:
            ar_g = _growth(c_in.accounts_receivable, p_in.accounts_receivable)
            if ar_g is not None and ar_g - rev_g > 15.0 and ar_g > 5.0:
                add("RECEIVABLES_DIVERGENCE", "medium", cur.fiscal_year, "ar_growth_minus_revenue_growth_pp",
                    ar_g - rev_g, 15.0,
                    f"RECEIVABLES DIVERGENCE (FY{cur.fiscal_year}): Accounts receivable grew {ar_g:+.2f}% "
                    f"(${p_in.accounts_receivable:,.1f}M -> ${c_in.accounts_receivable:,.1f}M) versus revenue "
                    f"growth of {rev_g:+.2f}%, which may signal aggressive revenue recognition or collection delays.")
        if p_in.inventories and c_in.inventories:
            inv_g = _growth(c_in.inventories, p_in.inventories)
            if inv_g is not None and inv_g - rev_g > 20.0 and inv_g > 10.0:
                add("INVENTORY_ACCUMULATION", "medium", cur.fiscal_year, "inventory_growth_minus_revenue_growth_pp",
                    inv_g - rev_g, 20.0,
                    f"INVENTORY ACCUMULATION WARNING (FY{cur.fiscal_year}): Inventory increased {inv_g:+.2f}% "
                    f"(${p_in.inventories:,.1f}M -> ${c_in.inventories:,.1f}M), substantially exceeding revenue "
                    f"growth of {rev_g:+.2f}%; this may indicate obsolescence risk or slowing sell-through.")

    for yr, q in eq.items():
        if q.non_operating_to_operating_income_pct is not None and \
                q.non_operating_to_operating_income_pct > NON_OPERATING_TO_EBIT_THRESHOLD_PCT:
            add("NON_CORE_INCOME_DEPENDENCE", "medium", yr, "non_operating_to_operating_income_pct",
                q.non_operating_to_operating_income_pct, NON_OPERATING_TO_EBIT_THRESHOLD_PCT,
                f"NON-CORE INCOME DEPENDENCE (FY{yr}): Non-operating income of ${q.non_operating_income:,.1f}M "
                f"equals {q.non_operating_to_operating_income_pct:.1f}% of operating income; pre-tax earnings "
                f"lean on interest and other income rather than the core business.")
        if q.adjustments_pct_of_operating_income is not None and \
                q.adjustments_pct_of_operating_income > NORMALIZATION_TO_EBIT_THRESHOLD_PCT:
            labels = ", ".join(a.label for a in q.adjustments if a.affects == "operating_income")
            add("NON_CORE_INCOME_DEPENDENCE", "high", yr, "normalization_adjustments_pct_of_operating_income",
                q.adjustments_pct_of_operating_income, NORMALIZATION_TO_EBIT_THRESHOLD_PCT,
                f"NORMALIZATION MATERIALITY (FY{yr}): Non-recurring / non-core items ({labels}) equal "
                f"{q.adjustments_pct_of_operating_income:.1f}% of reported operating income; normalized operating "
                f"income is ${q.normalized_operating_income:,.1f}M ({q.normalized_operating_margin_pct:.2f}% margin).")

        f_in = inputs[yr]
        tax_reasons = []
        severity: Severity = "medium"
        if f_in.income_tax_expense is not None and f_in.pretax_income is not None and f_in.pretax_income > 0:
            if f_in.income_tax_expense < 0:
                severity = "high"
                tax_reasons.append(
                    f"a ${abs(f_in.income_tax_expense):,.1f}M tax benefit on positive pre-tax income of "
                    f"${f_in.pretax_income:,.1f}M")
            elif q.effective_tax_rate_pct is not None and not (ETR_LOW_PCT <= q.effective_tax_rate_pct <= ETR_HIGH_PCT):
                tax_reasons.append(f"an effective tax rate of {q.effective_tax_rate_pct:.1f}%")
        if q.deferred_tax_to_net_income_pct is not None and \
                abs(q.deferred_tax_to_net_income_pct) > DEFERRED_TAX_TO_NI_THRESHOLD_PCT:
            tax_reasons.append(
                f"a deferred tax adjustment of ${f_in.deferred_income_taxes:,.1f}M "
                f"({q.deferred_tax_to_net_income_pct:.1f}% of net income)")
        if tax_reasons:
            add("TAX_ANOMALY", severity, yr, "effective_tax_rate_pct", q.effective_tax_rate_pct, None,
                f"TAX ANOMALY (FY{yr}): Reported earnings include {'; '.join(tax_reasons)}. Net income and margins "
                f"for this year are not comparable without normalization (e.g. valuation-allowance release).")

        if q.sbc_to_ocf_pct is not None and (
            q.sbc_to_ocf_pct > SBC_TO_OCF_THRESHOLD_PCT
            or (q.sbc_to_operating_income_pct or 0.0) > SBC_TO_EBIT_THRESHOLD_PCT
        ):
            add("SBC_INTENSITY", "medium", yr, "sbc_to_ocf_pct", q.sbc_to_ocf_pct, SBC_TO_OCF_THRESHOLD_PCT,
                f"STOCK-BASED COMPENSATION INTENSITY (FY{yr}): SBC of ${q.stock_based_compensation:,.1f}M equals "
                f"{q.sbc_to_ocf_pct:.1f}% of operating cash flow and "
                f"{q.sbc_to_operating_income_pct or 0.0:.1f}% of operating income; SBC-adjusted FCF is "
                f"${q.sbc_adjusted_fcf:,.1f}M.")

        if q.sloan_accruals_ratio_pct is not None and q.sloan_accruals_ratio_pct > SLOAN_ACCRUALS_THRESHOLD_PCT:
            add("HIGH_ACCRUALS", "medium", yr, "sloan_accruals_ratio_pct", q.sloan_accruals_ratio_pct,
                SLOAN_ACCRUALS_THRESHOLD_PCT,
                f"HIGH ACCRUALS (FY{yr}): Sloan accruals ratio of {q.sloan_accruals_ratio_pct:.1f}% of average total "
                f"assets; earnings are running materially ahead of operating cash flow.")

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

    for yr, c in cs.items():
        if c.opex_growth_minus_revenue_growth_pp is not None and \
                c.opex_growth_minus_revenue_growth_pp > OPEX_VS_REVENUE_GROWTH_THRESHOLD_PP:
            add("NEGATIVE_OPERATING_LEVERAGE", "medium", yr, "opex_growth_minus_revenue_growth_pp",
                c.opex_growth_minus_revenue_growth_pp, OPEX_VS_REVENUE_GROWTH_THRESHOLD_PP,
                f"NEGATIVE OPERATING LEVERAGE (FY{yr}): Operating expenses grew {c.operating_expense_growth_pct:+.2f}% "
                f"against revenue growth of {c.revenue_growth_pct:+.2f}% "
                f"({c.opex_growth_minus_revenue_growth_pp:.1f}pp gap).")

    margin_moves = [(yr, c.operating_margin_change_bps) for yr, c in sorted(cs.items())
                    if c.operating_margin_change_bps is not None]
    if margin_moves:
        worst_yr, worst = min(margin_moves, key=lambda m: m[1])
        cumulative = history[-1].operating_margin_pct - history[0].operating_margin_pct
        all_down = all(m[1] < 0 for m in margin_moves)
        if worst <= -MARGIN_SINGLE_YEAR_DROP_BPS or (all_down and cumulative * 100.0 <= -MARGIN_CUMULATIVE_DROP_BPS):
            add("MARGIN_COMPRESSION", "medium", history[-1].fiscal_year, "operating_margin_change_bps",
                round(cumulative * 100.0, 0), -MARGIN_CUMULATIVE_DROP_BPS,
                f"MARGIN COMPRESSION: Operating margin moved from {history[0].operating_margin_pct:.2f}% "
                f"(FY{history[0].fiscal_year}) to {history[-1].operating_margin_pct:.2f}% "
                f"(FY{history[-1].fiscal_year}), {cumulative * 100.0:,.0f} bps; worst single year FY{worst_yr} "
                f"({worst:,.0f} bps).")

    for yr, w in wc.items():
        for metric, change in (("dso", w.dso_change_pct), ("dio", w.dio_change_pct)):
            if change is not None and change > WORKING_CAPITAL_DAYS_THRESHOLD_PCT:
                days = w.dso_days if metric == "dso" else w.dio_days
                add("WORKING_CAPITAL_DETERIORATION", "low", yr, f"{metric}_change_pct", change,
                    WORKING_CAPITAL_DAYS_THRESHOLD_PCT,
                    f"WORKING CAPITAL DETERIORATION (FY{yr}): {metric.upper()} lengthened {change:.1f}% to "
                    f"{days:.1f} days.")

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
