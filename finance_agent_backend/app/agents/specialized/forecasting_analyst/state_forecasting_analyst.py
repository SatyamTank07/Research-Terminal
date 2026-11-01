"""Output schemas of the Forecasting Analyst: every number is code-built."""

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline
from app.agents.specialized.forecasting_analyst.modules.forecast_consistency import ForecastDiagnostics
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule import ForecastYearResult
from app.agents.specialized.forecasting_analyst.modules.management_guidance import MdnaAssessment
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import DriverAdjustment, ForecastDrivers
from app.agents.specialized.forecasting_analyst.modules.scenarios import ForecastScenario

ForecastYear = ForecastYearResult
GuidanceSource = Literal["mdna_informed", "historical_fade"]


class ForecastOutput(BaseModel):
    """Structured artifact emitted by the Financial Forecasting Analyst Agent. Every number is code-built."""

    ticker: str = Field(..., description="Stock ticker symbol")
    fiscal_year: int = Field(..., description="Base 10-K fiscal year")
    base_revenue: float = Field(..., description="Audited base-year revenue ($M)")
    forecast_horizon_years: int = Field(..., description="Explicit years modeled (5, or 10 when the growth fade was extended)")
    revenue_cagr_pct: float = Field(..., description="Base-case revenue CAGR over the explicit horizon (%)")
    cumulative_fcf: float = Field(..., description="Sum of base-case UFCF over the explicit horizon ($M)")
    average_annual_fcf: float = Field(..., description="Average base-case UFCF per year ($M)")
    guidance_source: GuidanceSource = Field(
        ..., description="'mdna_informed' = LLM drivers screened against Item 7 evidence; 'historical_fade' = no-LLM fade"
    )
    tax_rate_pct: float = Field(..., description="Tax rate applied to EBIT (%)")
    projected_fcfs: List[float] = Field(..., description="Base-case UFCF ($M; SBC already expensed in EBIT), handed to the DCF")
    forecast_schedule: List[ForecastYearResult]
    forecast_table_markdown: str
    assumptions: ForecastBaseline
    applied_drivers: ForecastDrivers
    driver_adjustments: List[DriverAdjustment] = Field(default_factory=list)
    scenarios: List[ForecastScenario]
    diagnostics: ForecastDiagnostics
    mdna_assessment: MdnaAssessment
    growth_rationale: str
    margin_rationale: str
    reinvestment_rationale: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)


class ForecastBuild(BaseModel):
    """The code-built forecast before assembly: applied drivers, base schedule, scenarios and diagnostics."""

    drivers: ForecastDrivers
    schedule: Dict[str, Any]
    scenarios: List[ForecastScenario]
    diagnostics: ForecastDiagnostics

    def preview_rows(self) -> List[Dict[str, float]]:
        return [
            {
                "year": y["projected_year"],
                "revenue": y["projected_revenue"],
                "growth_pct": y["projected_revenue_growth_pct"],
                "ebit_margin_pct": y["projected_ebit_margin_pct"],
                "capex_pct": round(y["projected_capex"] / y["projected_revenue"] * 100.0, 2)
                if y["projected_revenue"] else 0.0,
                "ufcf": y["projected_unlevered_fcf"],
            }
            for y in self.schedule["forecast_schedule"]
        ]
