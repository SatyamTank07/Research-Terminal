"""Result types of the market-implied-expectations module."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

ReverseDCFStatus = Literal[
    "solved", "price_above_range", "price_below_range", "no_solution", "price_unavailable", "forecast_unavailable"
]


@dataclass(frozen=True)
class MarketImpliedExpectations:
    """What the share price already assumes, against the base case (reverse DCF)."""

    market_implied_growth_pct: Optional[float]
    modeled_revenue_cagr_pct: Optional[float]
    reverse_dcf_horizon_years: Optional[int]
    market_implied_margin_shift_pp: Optional[float]
    modeled_final_margin_pct: Optional[float]
    reverse_dcf_status: ReverseDCFStatus
    reverse_dcf_summary: str
    market_implied_wacc_pct: Optional[float]
    warnings: Tuple[str, ...]
