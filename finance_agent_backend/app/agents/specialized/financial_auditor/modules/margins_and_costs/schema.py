"""One year of cost structure: opex growth against revenue growth, margin moves in bps."""

from typing import Optional

from pydantic import BaseModel


class YearCostStructure(BaseModel):
    fiscal_year: int
    operating_expenses: Optional[float] = None
    operating_expense_growth_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    opex_growth_minus_revenue_growth_pp: Optional[float] = None
    gross_margin_change_bps: Optional[float] = None
    operating_margin_change_bps: Optional[float] = None
