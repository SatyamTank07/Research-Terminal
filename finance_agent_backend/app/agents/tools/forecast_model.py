"""Forecast assembly: base schedule, scenarios and diagnostics."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import ForecastDrivers, extend_fade
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path import apply_reinvestment_path
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule import run_schedule
from app.agents.specialized.forecasting_analyst.modules.scenarios import ForecastScenario, build_scenarios

GROWTH_SUPPORT_GAP_WARNING = 0.05
RONIC_WARNING = 1.0
MIN_REINVESTMENT_SHARE_OF_REVENUE = 0.005


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


class ForecastBuild(BaseModel):
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


def build_forecast(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    terminal_growth: float,
    risk_profile: Optional[str] = None,
) -> ForecastBuild:
    growth, margins, extended = extend_fade(drivers.revenue_growth, drivers.operating_margin, terminal_growth)
    applied, reinvestment_warnings = apply_reinvestment_path(
        baseline, drivers.model_copy(update={"revenue_growth": growth, "operating_margin": margins}), terminal_growth
    )
    schedule = run_schedule(baseline, applied)
    diagnostics = compute_diagnostics(baseline, applied, schedule, terminal_growth, extended)
    diagnostics.warnings.extend(reinvestment_warnings)
    return ForecastBuild(
        drivers=applied,
        schedule=schedule,
        scenarios=build_scenarios(baseline, applied, schedule, risk_profile, terminal_growth),
        diagnostics=diagnostics,
    )


def compute_diagnostics(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    schedule: Dict[str, Any],
    terminal_growth: float,
    extended: bool,
) -> ForecastDiagnostics:
    years = schedule["forecast_schedule"]
    final = years[-1]
    final_growth = drivers.revenue_growth[-1]
    net_reinvestment = [
        y["projected_capex"] - (y["projected_depreciation"] or 0.0) + y["projected_nwc_change"] for y in years
    ]
    reinvestment_rate = net_reinvestment[-1] / final["projected_nopat"] if final["projected_nopat"] > 0 else None
    base_nopat = baseline.base_revenue * baseline.base_operating_margin * (1.0 - drivers.tax_rate)
    base_roic = baseline.base_roic if baseline.base_roic is not None and baseline.base_roic > 0 else None
    total_reinvestment = sum(net_reinvestment)
    total_revenue = sum(y["projected_revenue"] for y in years)
    ronic = (
        (final["projected_nopat"] - base_nopat) / total_reinvestment
        if total_reinvestment > MIN_REINVESTMENT_SHARE_OF_REVENUE * total_revenue
        else None
    )
    implied = reinvestment_rate * base_roic if reinvestment_rate is not None and base_roic is not None else None
    target_rate = (
        terminal_growth / baseline.steady_state_ronic if drivers.reinvestment_method == "value_driver_fade" else None
    )

    warnings = list(baseline.warnings)
    if reinvestment_rate is not None and reinvestment_rate < 0 and final_growth > 0:
        warnings.append(
            "Final-year net reinvestment is negative while revenue still grows; terminal free cash flow may be overstated."
        )
    if ronic is not None and ronic > RONIC_WARNING:
        warnings.append(
            f"Return on new invested capital of {ronic * 100:.0f}% exceeds 100%; CapEx or working-capital "
            "assumptions may understate the reinvestment growth requires."
        )
    if implied is not None and abs(implied - final_growth) > GROWTH_SUPPORT_GAP_WARNING:
        warnings.append(
            f"Reinvestment supports {implied * 100:.2f}% growth versus {final_growth * 100:.2f}% modeled in the final year."
        )
    return ForecastDiagnostics(
        final_year_growth_pct=round(final_growth * 100.0, 2),
        terminal_growth_pct=round(terminal_growth * 100.0, 2),
        fade_extended=extended,
        final_year_reinvestment_rate_pct=_pct(reinvestment_rate),
        base_roic_pct=_pct(base_roic),
        ronic_pct=_pct(ronic),
        steady_state_ronic_pct=_pct(baseline.steady_state_ronic),
        target_reinvestment_rate_pct=_pct(target_rate),
        implied_fundamental_growth_pct=_pct(implied),
        warnings=warnings,
    )


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value * 100.0, 2)
