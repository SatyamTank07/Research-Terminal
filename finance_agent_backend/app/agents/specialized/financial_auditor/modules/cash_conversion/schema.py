"""One year of cash conversion: accruals and stock-based compensation against cash flow."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CashConversionYear:
    fiscal_year: int
    accruals: float
    sloan_accruals_ratio_pct: Optional[float]
    stock_based_compensation: Optional[float]
    sbc_to_ocf_pct: Optional[float]
    sbc_to_operating_income_pct: Optional[float]
    sbc_adjusted_fcf: Optional[float]
