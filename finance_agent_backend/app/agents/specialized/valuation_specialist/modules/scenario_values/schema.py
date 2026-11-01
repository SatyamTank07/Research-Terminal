"""Result types of the scenario-values module."""

from dataclasses import dataclass
from typing import List, Literal, Optional

from pydantic import BaseModel


class ScenarioValuation(BaseModel):
    name: Literal["bull", "base", "bear"]
    probability: float
    revenue_cagr_pct: Optional[float] = None
    cumulative_fcf: float
    enterprise_value: float
    fair_value_per_share: float
    upside_downside_pct: Optional[float] = None


@dataclass(frozen=True)
class ScenarioValues:
    """Fair value per bull / base / bear scenario, the probability-weighted value and the scenario table."""

    scenario_valuations: List[ScenarioValuation]
    probability_weighted_fair_value: Optional[float]
    bear_fair_value: Optional[float]
    bull_fair_value: Optional[float]
    scenario_table_markdown: str
