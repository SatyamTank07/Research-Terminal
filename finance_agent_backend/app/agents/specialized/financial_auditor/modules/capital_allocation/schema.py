"""Returns on capital, capital intensity and the share count, per audited year."""

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class CapitalReturns:
    """Latest-year NOPAT, invested capital and the returns on it."""

    nopat: float
    invested_capital: float
    capital_basis: Literal["average", "ending"]
    roic_pct: Optional[float]
    roic_on_gross_capital_pct: Optional[float]
    roe_pct: Optional[float]


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
