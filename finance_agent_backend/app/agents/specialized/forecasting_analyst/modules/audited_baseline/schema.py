"""Result types of the audited-baseline module."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class HistoricalRow(BaseModel):
    """Audited year shown as an 'A' column in the forecast table (percentages are whole percents, display only)."""

    fiscal_year: int
    revenue: float
    revenue_growth_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None


class ForecastBaseline(BaseModel):
    """Audited starting point and reinvestment intensities for the forecast (ratios are decimals)."""

    base_year: int
    base_revenue: float = Field(..., description="Base-year revenue ($M)")
    base_operating_margin: float = Field(..., description="Base-year EBIT margin (decimal)")
    margin_source: Literal["normalized", "reported"]
    latest_revenue_growth: Optional[float] = None
    historical_cagr: Optional[float] = None
    history_years: int
    growth_volatility: Optional[float] = Field(None, description="Population std-dev of yearly revenue growth")
    tax_rate: float
    tax_source: Literal["median_effective", "statutory_default"]
    capex_pct_of_revenue: float = Field(
        ..., description="Starting (year-1) CapEx intensity: recency-weighted (1:2:3) average of the audited years"
    )
    capex_average_pct_of_revenue: float = Field(..., description="Simple average CapEx intensity, same years")
    latest_capex_pct_of_revenue: float = Field(..., description="Base-year CapEx / revenue")
    capex_years: int
    da_pct_of_revenue: float
    da_source: Literal["reported", "ebitda_bridge", "capex_parity_proxy"]
    nwc_pct_of_incremental_revenue: float
    nwc_source: Literal["working_capital_days", "not_available"]
    sbc_pct_of_revenue: float
    sbc_source: Literal["reported", "not_reported"]
    base_invested_capital: Optional[float] = Field(None, description="Total debt + stockholders' equity ($M)")
    base_roic: Optional[float] = Field(
        None, description="Base-year NOPAT / invested capital (decimal); None when invested capital is not positive"
    )
    steady_state_ronic: float = Field(
        ..., description="Return on new invested capital that sizes final-year net reinvestment (decimal)"
    )
    steady_state_ronic_source: Literal[
        "base_roic", "base_roic_floored", "base_roic_capped", "default_no_positive_roic"
    ]
    margin_trend: Literal["expanding", "contracting", "mixed", "insufficient_history"] = Field(
        ..., description="Direction of the last three audited (reported) operating margins"
    )
    historical_average_margin: Optional[float] = Field(
        None, description="Mean audited operating margin over the history (decimal); None with one year"
    )
    historical_rows: List[HistoricalRow] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
