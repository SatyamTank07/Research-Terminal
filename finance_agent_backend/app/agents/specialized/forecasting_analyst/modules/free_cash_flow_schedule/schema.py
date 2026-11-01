"""Result types of the free-cash-flow-schedule module ($ Millions; percentages are whole percents)."""

from typing import List

from pydantic import BaseModel, Field


class ForecastYearResult(BaseModel):
    """Projected line items for one forward year ($ Millions; percentages are whole percents)."""

    projected_year: int
    projected_revenue: float
    projected_revenue_growth_pct: float
    projected_ebit: float
    projected_ebit_margin_pct: float
    projected_nopat: float
    projected_capex: float
    projected_depreciation: float
    projected_nwc_change: float = 0.0
    projected_sbc: float = Field(0.0, description="Stock-based compensation (memo: already expensed in EBIT)")
    projected_unlevered_fcf: float


class ForecastScheduleResult(BaseModel):
    base_revenue: float
    base_year: int
    forecast_horizon_years: int
    revenue_cagr_pct: float
    cumulative_fcf: float
    average_annual_fcf: float
    tax_rate_pct: float
    projected_fcfs: List[float]
    forecast_schedule: List[ForecastYearResult]
    forecast_table_markdown: str
