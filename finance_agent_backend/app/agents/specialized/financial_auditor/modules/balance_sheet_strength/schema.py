"""The latest balance sheet (liquidity bridge, debt, share count) and the solvency ratios built on it."""

from typing import Optional

from pydantic import BaseModel, Field


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


class SolvencyRatiosResult(BaseModel):
    ebitda: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    net_debt_to_ebitda_interpretation: str
    current_ratio: Optional[float] = None
    total_debt_to_equity: Optional[float] = None
