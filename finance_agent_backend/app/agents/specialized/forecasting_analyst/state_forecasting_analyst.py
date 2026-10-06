"""State and Output Schemas for Financial Forecasting Analyst Agent."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.agents.tools.forecast_tools import (
    ForecastYearResult,
    ForecastScheduleResult,
    ProvenanceMode,
)

ForecastYear = ForecastYearResult
GuidanceSource = Literal["md&a_explicit", "historical_cagr_decay"]


class ForecastOutput(BaseModel):
    """Structured artifact emitted by the Financial Forecasting Analyst Agent."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="Base 10-K fiscal year audited (e.g. 2025)")
    base_revenue: float = Field(..., description="Audited base year revenue in $ Millions")
    forecast_horizon_years: int = Field(default=5, description="Number of forward explicit projection years")
    revenue_cagr_pct: float = Field(..., description="5-year Compound Annual Growth Rate of Revenue as %")
    cumulative_5yr_fcf: float = Field(..., description="Total cumulative explicit UFCF generated over 5 years in $ Millions")
    average_annual_fcf: float = Field(..., description="Average annual UFCF in $ Millions")
    provenance_mode: ProvenanceMode = Field(
        ..., description="Accounting bridge used: 'comprehensive_line_item' or 'simplified_nopat_less_capex'"
    )
    guidance_source: GuidanceSource = Field(
        default="md&a_explicit",
        description="Provenance of forward growth assumptions: 'md&a_explicit' (derived from 10-K Item 7 guidance) or 'historical_cagr_decay' (deterministic 75bps/yr decay from historical 3-year CAGR)",
    )
    tax_rate_pct: float = Field(..., description="Effective tax rate percentage used")

    # Direct handoff into DCF Valuation Specialist
    projected_fcfs: List[float] = Field(
        ..., description="Explicit forecast period UFCFs ($M) feeding directly into calculate_dcf_tool"
    )
    forecast_schedule: List[ForecastYearResult] = Field(
        ..., description="Year-by-year projected line items (Revenue, EBIT, NOPAT, CapEx, UFCF)"
    )
    forecast_table_markdown: str = Field(
        ..., description="Publication-ready Markdown forecast comparison table"
    )

    # Forecaster Synthesis & Rationale
    growth_rationale: str = Field(
        ..., description="Evidence-backed defense of revenue growth curve linked to Item 7 MD&A disclosures"
    )
    margin_expansion_rationale: str = Field(
        ..., description="Rationale for EBIT margin progression (operating leverage, product mix, cost discipline)"
    )
    reinvestment_rationale: Optional[str] = Field(
        None, description="Analysis of CapEx and working capital intensity relative to historical trend"
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Citations to Item 7 tables and MD&A disclosures"
    )
