"""What did the last 3-5 years look like? Revenue, margins, operating cash flow, capex, FCF and growth per year."""

import math
from typing import Any, List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.financial_history.schema import YearFinancialsResult


def require_finite(val: Any, name: str) -> float:
    if val is None:
        raise ValueError(f"{name} cannot be None.")
    if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        raise ValueError(f"{name} must be a valid finite number, got: {val}")
    return float(val)


def pct_of(num: Optional[float], den: Optional[float], digits: int = 2) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return round(num / den * 100.0, digits)


def growth_pct(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    if cur is None or prev is None or prev == 0:
        return None
    return round((cur - prev) / abs(prev) * 100.0, 2)


def cost_of_revenue_of(inp: AnnualFinancialInput) -> Optional[float]:
    if inp.cost_of_revenue is not None:
        return inp.cost_of_revenue
    if inp.gross_profit is not None:
        return inp.revenue - inp.gross_profit
    return None


def gross_profit_of(inp: AnnualFinancialInput) -> Optional[float]:
    if inp.gross_profit is not None:
        return inp.gross_profit
    if inp.cost_of_revenue is not None:
        return round(inp.revenue - inp.cost_of_revenue, 4)
    return None


def operating_expenses_of(inp: AnnualFinancialInput) -> Optional[float]:
    if inp.operating_expenses is not None:
        return inp.operating_expenses
    gp = gross_profit_of(inp)
    return round(gp - inp.operating_income, 4) if gp is not None else None


def build_financial_history(annual_financials: List[AnnualFinancialInput]) -> List[YearFinancialsResult]:
    """One row per audited year, oldest first; growth compares with the year before."""
    sorted_years = sorted(annual_financials, key=lambda x: x.fiscal_year)
    history: List[YearFinancialsResult] = []
    prev: Optional[AnnualFinancialInput] = None
    prev_fcf: Optional[float] = None

    for f in sorted_years:
        rev = require_finite(f.revenue, f"revenue ({f.fiscal_year})")
        ebit = require_finite(f.operating_income, f"operating_income ({f.fiscal_year})")
        ni = require_finite(f.net_income, f"net_income ({f.fiscal_year})")
        ocf = require_finite(f.operating_cash_flow, f"operating_cash_flow ({f.fiscal_year})")
        capex = require_finite(f.capital_expenditures, f"capital_expenditures ({f.fiscal_year})")
        gp = gross_profit_of(f)
        fcf = round(ocf - capex, 2)

        history.append(YearFinancialsResult(
            fiscal_year=f.fiscal_year,
            revenue=rev,
            gross_profit=gp,
            gross_margin_pct=pct_of(gp, rev),
            operating_income=ebit,
            operating_margin_pct=pct_of(ebit, rev) or 0.0,
            net_income=ni,
            net_margin_pct=pct_of(ni, rev) or 0.0,
            operating_cash_flow=ocf,
            capital_expenditures=capex,
            free_cash_flow=fcf,
            depreciation_amortization=f.depreciation_amortization,
            interest_expense=abs(f.interest_expense) if f.interest_expense is not None else None,
            fcf_conversion_pct=pct_of(fcf, ni),
            revenue_growth_pct=growth_pct(rev, prev.revenue) if prev else None,
            operating_income_growth_pct=growth_pct(ebit, prev.operating_income) if prev else None,
            fcf_growth_pct=growth_pct(fcf, prev_fcf) if prev else None,
        ))
        prev, prev_fcf = f, fcf
    return history
