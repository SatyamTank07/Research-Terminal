"""Result types of the scenarios module."""

from typing import List, Literal

from pydantic import BaseModel


class ForecastScenario(BaseModel):
    name: Literal["bull", "base", "bear"]
    probability: float
    revenue_growth_pct: List[float]
    operating_margin_pct: List[float]
    projected_fcfs: List[float]
    revenue_cagr_pct: float
    cumulative_fcf: float
