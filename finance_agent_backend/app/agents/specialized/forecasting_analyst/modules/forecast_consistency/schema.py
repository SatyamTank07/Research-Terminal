"""Result types of the forecast-consistency module."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ForecastDiagnostics(BaseModel):
    final_year_growth_pct: Optional[float] = None
    terminal_growth_pct: Optional[float] = None
    fade_extended: bool = False
    final_year_reinvestment_rate_pct: Optional[float] = Field(
        None, description="(CapEx - D&A + dNWC) / NOPAT in the final explicit year"
    )
    base_roic_pct: Optional[float] = Field(None, description="Base-year NOPAT / (total debt + equity); None if <= 0")
    ronic_pct: Optional[float] = Field(
        None, description="Change in NOPAT / cumulative net reinvestment (None when reinvestment is negligible)"
    )
    steady_state_ronic_pct: Optional[float] = Field(
        None, description="RONIC that sizes final-year reinvestment (base ROIC clamped to 8%-30%, else 10%)"
    )
    target_reinvestment_rate_pct: Optional[float] = Field(
        None,
        description="Terminal growth / steady-state RONIC: final-year net reinvestment as % of NOPAT "
        "(None under maintenance_fade)",
    )
    implied_fundamental_growth_pct: Optional[float] = Field(
        None, description="Final-year reinvestment rate x base-year ROIC"
    )
    warnings: List[str] = Field(default_factory=list)
