"""Result types of the revenue-and-margin-drivers module (ForecastDrivers is in decimals, the proposal in WHOLE percent)."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ForecastDrivers(BaseModel):
    """Per-year growth and margin paths plus reinvestment intensities (all decimals).

    `capex_pct_of_revenue` is the year-1 intensity; `reinvestment_path.apply_reinvestment_path` fills `capex_path`, which fades it
    linearly to `steady_state_capex_pct_of_revenue` in the final explicit year. D&A is held at
    `da_pct_of_revenue` (no PP&E roll-forward is modeled; the final-year net reinvestment is pinned by the
    steady state, so the D&A level does not change terminal cash flow).
    """

    revenue_growth: List[float]
    operating_margin: List[float]
    tax_rate: float
    capex_pct_of_revenue: float = Field(..., description="Year-1 CapEx intensity (decimal)")
    da_pct_of_revenue: float
    nwc_pct_of_incremental_revenue: float
    sbc_pct_of_revenue: float
    capex_path: List[float] = Field(
        default_factory=list, description="Applied per-year CapEx intensity (decimals); filled by reinvestment_path.apply_reinvestment_path"
    )
    steady_state_capex_pct_of_revenue: Optional[float] = Field(
        None, description="Final-year CapEx intensity consistent with terminal growth (decimal)"
    )
    reinvestment_method: Optional[Literal["value_driver_fade", "maintenance_fade"]] = Field(
        None,
        description=(
            "'value_driver_fade': final-year net reinvestment = terminal growth / steady-state RONIC x NOPAT; "
            "'maintenance_fade': final-year NOPAT is not positive, so CapEx fades to D&A"
        ),
    )


class ForecastDriverProposal(BaseModel):
    """Drivers submitted by the LLM, in WHOLE percent (6.5 = 6.5%)."""

    revenue_growth_pct: List[float]
    operating_margin_pct: List[float]
    margin_citation_chunk_id: Optional[str] = None
    capex_pct_of_revenue: Optional[float] = None
    capex_citation_chunk_id: Optional[str] = None
    nwc_pct_of_incremental_revenue: Optional[float] = None
    nwc_citation_chunk_id: Optional[str] = None
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class DriverAdjustment(BaseModel):
    driver: str
    year: Optional[int] = Field(None, description="1-based forecast year; None for single-value drivers")
    proposed: float = Field(..., description="Proposed value (decimal)")
    applied: float = Field(..., description="Value used (decimal)")
    action: Literal["clamped", "rejected"]
    reason: str


class DriverScreenResult(BaseModel):
    accepted: bool
    drivers: Optional[ForecastDrivers] = None
    adjustments: List[DriverAdjustment] = Field(default_factory=list)
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    margin_evidence_accepted: bool = Field(
        False, description="True when margin_citation_chunk_id is an MD&A chunk with forward-looking margin/cost language"
    )
    errors: List[str] = Field(default_factory=list)
