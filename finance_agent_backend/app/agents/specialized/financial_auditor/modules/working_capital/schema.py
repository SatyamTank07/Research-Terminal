"""One year of working-capital days (ending-balance basis)."""

from typing import Optional

from pydantic import BaseModel


class YearWorkingCapital(BaseModel):
    fiscal_year: int
    dso_days: Optional[float] = None
    dio_days: Optional[float] = None
    dpo_days: Optional[float] = None
    cash_conversion_cycle_days: Optional[float] = None
    dso_change_pct: Optional[float] = None
    dio_change_pct: Optional[float] = None
