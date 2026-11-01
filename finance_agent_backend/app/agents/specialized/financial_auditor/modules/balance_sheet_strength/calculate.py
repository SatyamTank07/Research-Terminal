"""Can it pay its debts? Liquidity bridge, net debt, the valuation share count, leverage and book equity."""

import logging
from typing import List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
)
from app.agents.specialized.financial_auditor.findings import ForensicFinding, add_finding
from app.agents.specialized.financial_auditor.modules.balance_sheet_strength.schema import (
    BalanceSheetResult,
    SolvencyRatiosResult,
)
from app.agents.specialized.financial_auditor.modules.financial_history import YearFinancialsResult

logger = logging.getLogger("finance_agent.agents.financial_auditor")


def measure_balance_sheet(
    balance_sheet: BalanceSheetInput,
    prior_balance_sheet: Optional[PriorBalanceSheetInput] = None,
) -> BalanceSheetResult:
    """Liquid cash, gross and net debt, and the share count per-share value is divided by."""
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

    return BalanceSheetResult(
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
        stockholders_equity=balance_sheet.stockholders_equity,
        noncontrolling_interests=balance_sheet.noncontrolling_interests,
        valuation_shares_outstanding=valuation_shares,
        share_count_source=share_source,
        weighted_diluted_shares=wd,
        period_end_shares_outstanding=pe,
        current_assets=balance_sheet.current_assets,
        current_liabilities=balance_sheet.current_liabilities,
        total_assets=balance_sheet.total_assets,
    )


def classify_book_equity(equity: float) -> str:
    """Positive book equity, or why equity-based ratios (ROIC, ROE, D/E) are suppressed."""
    if equity < 0:
        logger.warning(
            f"calculate_financial_ratios: Stockholders' equity is negative (${equity:,.1f}M). "
            f"Suppressing ROIC, ROE, and Debt/Equity."
        )
        return "negative_book_equity_ratios_suppressed"
    if equity == 0:
        return "zero_book_equity_ratios_suppressed"
    return "positive_book_equity"


def measure_solvency(
    balance: BalanceSheetResult,
    latest: YearFinancialsResult,
    latest_input: AnnualFinancialInput,
    equity_status: str,
) -> SolvencyRatiosResult:
    """EBITDA, net debt / EBITDA, the current ratio and debt / equity (only on positive book equity)."""
    total_debt_equity = None
    if equity_status == "positive_book_equity":
        total_debt_equity = round(balance.total_debt / balance.stockholders_equity, 2)

    ebitda = net_debt_ebitda = None
    ebitda_interp = "ebitda_not_available"
    if latest_input.depreciation_amortization is not None:
        ebitda = round(latest.operating_income + latest_input.depreciation_amortization, 2)
        if ebitda > 0:
            net_debt_ebitda = round(balance.net_debt / ebitda, 2)
            ebitda_interp = "net_cash_surplus" if balance.net_cash_position else "leverage_ratio"
        elif ebitda < 0:
            ebitda_interp = "negative_ebitda"

    current_ratio = None
    if balance.current_assets is not None and balance.current_liabilities:
        current_ratio = round(balance.current_assets / balance.current_liabilities, 2)

    return SolvencyRatiosResult(
        ebitda=ebitda,
        net_debt_to_ebitda=net_debt_ebitda,
        net_debt_to_ebitda_interpretation=ebitda_interp,
        current_ratio=current_ratio,
        total_debt_to_equity=total_debt_equity,
    )


def screen_balance_sheet_strength(balance_sheet: Optional[BalanceSheetInput]) -> List[ForensicFinding]:
    """NEGATIVE_BOOK_EQUITY: liabilities exceed assets, so equity-based ratios mean nothing."""
    findings: List[ForensicFinding] = []
    if balance_sheet is not None and balance_sheet.stockholders_equity < 0:
        add_finding(findings, "NEGATIVE_BOOK_EQUITY", "high", balance_sheet.fiscal_year, "stockholders_equity",
            balance_sheet.stockholders_equity, 0.0,
            f"NEGATIVE BOOK EQUITY WARNING: Stockholders' equity is negative "
            f"(${balance_sheet.stockholders_equity:,.1f}M). ROE, ROIC and Debt-to-Equity are suppressed as "
            f"uninterpretable.")
    return findings
