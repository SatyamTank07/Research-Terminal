"""Does management earn more than the capital costs, and where does the cash go? Returns, reinvestment, dilution."""

from typing import List, Literal, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput, PriorBalanceSheetInput
from app.agents.specialized.financial_auditor.findings import ForensicFinding, add_finding
from app.agents.specialized.financial_auditor.modules.balance_sheet_strength import BalanceSheetResult
from app.agents.specialized.financial_auditor.modules.capital_allocation.schema import (
    CapitalReturns,
    YearCapitalIntensity,
    YearShareCount,
)
from app.agents.specialized.financial_auditor.modules.capital_allocation.thresholds import (
    CAPEX_DRIVEN_FCF_THRESHOLD_PCT,
    CAPEX_TO_DEPRECIATION_FLOOR_X,
    DILUTED_SHARE_GROWTH_THRESHOLD_PCT,
    PERIOD_END_SHARE_GROWTH_THRESHOLD_PCT,
)
from app.agents.specialized.financial_auditor.modules.financial_history import (
    YearFinancialsResult,
    growth_pct,
    pct_of,
)


def measure_capital_returns(
    latest: YearFinancialsResult,
    balance: BalanceSheetResult,
    prior_balance_sheet: Optional[PriorBalanceSheetInput],
    tax_rate: float,
    equity_status: str,
) -> CapitalReturns:
    """NOPAT over invested capital (net of liquid cash) and over gross capital, and ROE, on average capital when possible."""
    total_debt, total_liquid = balance.total_debt, balance.total_liquid_cash
    nopat = round(latest.operating_income * (1.0 - tax_rate), 2)
    equity = balance.stockholders_equity
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

    roic_pct = roic_gross_pct = roe_pct = None
    if equity_status == "positive_book_equity":
        if invested_capital > 0:
            roic_pct = round(nopat / invested_capital * 100.0, 2)
        if gross_capital > 0:
            roic_gross_pct = round(nopat / gross_capital * 100.0, 2)
        roe_pct = round(latest.net_income / avg_equity * 100.0, 2)

    return CapitalReturns(
        nopat=nopat,
        invested_capital=invested_capital,
        capital_basis=capital_basis,
        roic_pct=roic_pct,
        roic_on_gross_capital_pct=roic_gross_pct,
        roe_pct=roe_pct,
    )


def measure_capital_intensity(
    annual_financials: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
) -> List[YearCapitalIntensity]:
    """Capex against revenue and D&A, and how much of each year's FCF change came from capex, per year."""
    hist_map = {h.fiscal_year: h for h in history}
    rows: List[YearCapitalIntensity] = []
    prev: Optional[AnnualFinancialInput] = None
    for f in sorted(annual_financials, key=lambda x: x.fiscal_year):
        h = hist_map[f.fiscal_year]
        rev = f.revenue
        ocf_chg = capex_chg = fcf_chg = from_capex = None
        if prev is not None:
            prev_h = hist_map[prev.fiscal_year]
            ocf_chg = round(f.operating_cash_flow - prev.operating_cash_flow, 2)
            capex_chg = round(f.capital_expenditures - prev.capital_expenditures, 2)
            fcf_chg = round(h.free_cash_flow - prev_h.free_cash_flow, 2)
            from_capex = pct_of(-capex_chg, fcf_chg) if fcf_chg else None
        rows.append(YearCapitalIntensity(
            fiscal_year=f.fiscal_year,
            capex_to_revenue_pct=pct_of(f.capital_expenditures, rev),
            capex_to_depreciation=round(f.capital_expenditures / f.depreciation_amortization, 2)
            if f.depreciation_amortization else None,
            ocf_change=ocf_chg,
            capex_change=capex_chg,
            fcf_change=fcf_chg,
            fcf_change_from_capex_pct=from_capex,
        ))
        prev = f
    return rows


def measure_share_count(annual_financials: List[AnnualFinancialInput]) -> List[YearShareCount]:
    """Diluted weighted and period-end share counts and their growth, per year."""
    rows: List[YearShareCount] = []
    prev: Optional[AnnualFinancialInput] = None
    for f in sorted(annual_financials, key=lambda x: x.fiscal_year):
        rows.append(YearShareCount(
            fiscal_year=f.fiscal_year,
            diluted_weighted_shares=f.diluted_weighted_shares,
            diluted_share_growth_pct=growth_pct(f.diluted_weighted_shares, prev.diluted_weighted_shares) if prev else None,
            period_end_shares_outstanding=f.period_end_shares_outstanding,
            period_end_share_growth_pct=growth_pct(f.period_end_shares_outstanding, prev.period_end_shares_outstanding)
            if prev else None,
        ))
        prev = f
    return rows


def screen_capital_allocation(
    history: List[YearFinancialsResult],
    capital_intensity: List[YearCapitalIntensity],
    share_count: List[YearShareCount],
) -> List[ForensicFinding]:
    """CAPEX_DRIVEN_FCF (FCF up only because capex fell), UNDERINVESTMENT and SHARE_DILUTION."""
    findings: List[ForensicFinding] = []
    ci = {r.fiscal_year: r for r in capital_intensity}
    sh = {r.fiscal_year: r for r in share_count}
    for yr, c in ci.items():
        if c.fcf_change is not None and c.fcf_change > 0 and (c.ocf_change or 0.0) <= 0 \
                and (c.capex_change or 0.0) < 0 and (c.fcf_change_from_capex_pct or 0.0) > CAPEX_DRIVEN_FCF_THRESHOLD_PCT:
            add_finding(findings, "CAPEX_DRIVEN_FCF", "medium", yr, "fcf_change_from_capex_pct", c.fcf_change_from_capex_pct,
                CAPEX_DRIVEN_FCF_THRESHOLD_PCT,
                f"CAPEX-DRIVEN FCF (FY{yr}): Free cash flow rose ${c.fcf_change:,.1f}M while operating cash flow "
                f"changed ${c.ocf_change:,.1f}M; a ${abs(c.capex_change):,.1f}M capex reduction explains "
                f"{c.fcf_change_from_capex_pct:.1f}% of the improvement, so it reflects lower reinvestment rather "
                f"than stronger cash generation.")

    capex_da = [ci[h.fiscal_year].capex_to_depreciation for h in history[-2:]]
    if len(capex_da) == 2 and all(v is not None and v < CAPEX_TO_DEPRECIATION_FLOOR_X for v in capex_da):
        add_finding(findings, "UNDERINVESTMENT", "info", history[-1].fiscal_year, "capex_to_depreciation", capex_da[-1],
            CAPEX_TO_DEPRECIATION_FLOOR_X,
            f"UNDERINVESTMENT SIGNAL: Capex has run below depreciation for two consecutive years "
            f"(latest {capex_da[-1]:.2f}x), implying a shrinking asset base.")

    for yr, s in sh.items():
        if s.diluted_share_growth_pct is not None and s.diluted_share_growth_pct > DILUTED_SHARE_GROWTH_THRESHOLD_PCT:
            add_finding(findings, "SHARE_DILUTION", "medium", yr, "diluted_share_growth_pct", s.diluted_share_growth_pct,
                DILUTED_SHARE_GROWTH_THRESHOLD_PCT,
                f"SHARE DILUTION (FY{yr}): Weighted diluted shares grew {s.diluted_share_growth_pct:.2f}% "
                f"to {s.diluted_weighted_shares:,.1f}M.")
        if s.period_end_share_growth_pct is not None and \
                s.period_end_share_growth_pct > PERIOD_END_SHARE_GROWTH_THRESHOLD_PCT:
            add_finding(findings, "SHARE_DILUTION", "high", yr, "period_end_share_growth_pct", s.period_end_share_growth_pct,
                PERIOD_END_SHARE_GROWTH_THRESHOLD_PCT,
                f"SHARE COUNT EXPANSION (FY{yr}): Shares outstanding at year end rose "
                f"{s.period_end_share_growth_pct:.2f}% to {s.period_end_shares_outstanding:,.1f}M; investigate "
                f"equity awards, conversions or issuances, and use the period-end count for per-share value.")
    return findings
