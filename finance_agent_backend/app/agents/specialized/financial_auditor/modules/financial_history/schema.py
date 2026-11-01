"""One audited year of reported results ($ millions; margins and growth in percent)."""

from typing import Optional

from pydantic import BaseModel


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
