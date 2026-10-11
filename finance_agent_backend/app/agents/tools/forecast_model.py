"""Forecast assembly: fade extension, base schedule, bull/base/bear scenarios and reinvestment diagnostics."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.agents.tools.forecast_baseline import ForecastBaseline
from app.agents.tools.forecast_drivers import ForecastDrivers, extend_fade
from app.agents.tools.forecast_tools import calculate_forecast_schedule

SCENARIO_PROBABILITIES = {"bull": 0.25, "base": 0.50, "bear": 0.25}
SCENARIO_ORDER = {"bull": 0, "base": 1, "bear": 2}
MIN_SCENARIO_GROWTH_SHIFT = 0.02
MAX_SCENARIO_GROWTH_SHIFT = 0.10
BULL_MARGIN_SHIFT = 0.01
BEAR_MARGIN_SHIFT = -0.015
RISK_MULTIPLIERS = {"High": 1.5, "Moderate": 1.0, "Low": 0.75}
GROWTH_SUPPORT_GAP_WARNING = 0.05
RONIC_WARNING = 1.0
MIN_REINVESTMENT_SHARE_OF_REVENUE = 0.005


class ForecastScenario(BaseModel):
    name: Literal["bull", "base", "bear"]
    probability: float
    revenue_growth_pct: List[float]
    operating_margin_pct: List[float]
    projected_fcfs: List[float]
    revenue_cagr_pct: float
    cumulative_fcf: float


class ForecastDiagnostics(BaseModel):
    final_year_growth_pct: Optional[float] = None
    terminal_growth_pct: Optional[float] = None
    fade_extended: bool = False
    final_year_reinvestment_rate_pct: Optional[float] = Field(
        None, description="(CapEx - D&A + dNWC) / NOPAT in the final explicit year"
    )
    base_roic_pct: Optional[float] = Field(None, description="Base-year NOPAT / (total debt + equity)")
    ronic_pct: Optional[float] = Field(
        None, description="Change in NOPAT / cumulative net reinvestment (None when reinvestment is negligible)"
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
                "ufcf": y["projected_unlevered_fcf"],
            }
            for y in self.schedule["forecast_schedule"]
        ]


def run_schedule(baseline: ForecastBaseline, drivers: ForecastDrivers) -> Dict[str, Any]:
    return calculate_forecast_schedule(
        base_revenue=baseline.base_revenue,
        base_year=baseline.base_year,
        revenue_growth_rates=drivers.revenue_growth,
        operating_margins=drivers.operating_margin,
        tax_rate=drivers.tax_rate,
        capex_pct_of_revenue=drivers.capex_pct_of_revenue,
        depreciation_pct_of_revenue=drivers.da_pct_of_revenue,
        nwc_change_pct_of_revenue=drivers.nwc_pct_of_incremental_revenue,
        sbc_pct_of_revenue=drivers.sbc_pct_of_revenue,
        historical_rows=baseline.historical_rows,
    )


def build_forecast(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    terminal_growth: float,
    risk_profile: Optional[str] = None,
) -> ForecastBuild:
    growth, margins, extended = extend_fade(drivers.revenue_growth, drivers.operating_margin, terminal_growth)
    applied = drivers.model_copy(update={"revenue_growth": growth, "operating_margin": margins})
    schedule = run_schedule(baseline, applied)
    return ForecastBuild(
        drivers=applied,
        schedule=schedule,
        scenarios=build_scenarios(baseline, applied, schedule, risk_profile),
        diagnostics=compute_diagnostics(baseline, applied, schedule, terminal_growth, extended),
    )


def build_scenarios(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    base_schedule: Dict[str, Any],
    risk_profile: Optional[str],
) -> List[ForecastScenario]:
    """Growth shifts are largest in year 1 and vanish by the final year, so every case meets terminal growth.

    Bull and bear are defined by free cash flow: when the base case burns cash, faster growth burns more,
    so the growth shift flips sign while the margin shift keeps its direction.
    """
    shift = min(max(baseline.growth_volatility or 0.0, MIN_SCENARIO_GROWTH_SHIFT), MAX_SCENARIO_GROWTH_SHIFT)
    if base_schedule["cumulative_fcf"] < 0:
        shift = -shift
    risk_mult = RISK_MULTIPLIERS.get(risk_profile or "", 1.0)
    n = len(drivers.revenue_growth)
    weights = [1.0] if n == 1 else [(n - 1 - i) / (n - 1) for i in range(n)]
    variants = {
        "bull": (shift, BULL_MARGIN_SHIFT),
        "bear": (-shift * risk_mult, BEAR_MARGIN_SHIFT * risk_mult),
    }
    scenarios = [_scenario("base", drivers, base_schedule)]
    for name, (g_shift, m_shift) in variants.items():
        variant = drivers.model_copy(update={
            "revenue_growth": [round(g + g_shift * w, 4) for g, w in zip(drivers.revenue_growth, weights)],
            "operating_margin": [round(min(max(m + m_shift, -2.0), 1.0), 4) for m in drivers.operating_margin],
        })
        scenarios.append(_scenario(name, variant, run_schedule(baseline, variant)))
    return sorted(scenarios, key=lambda s: SCENARIO_ORDER[s.name])


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
    invested = baseline.base_invested_capital
    base_roic = base_nopat / invested if invested and base_nopat > 0 else None
    total_reinvestment = sum(net_reinvestment)
    total_revenue = sum(y["projected_revenue"] for y in years)
    ronic = (
        (final["projected_nopat"] - base_nopat) / total_reinvestment
        if total_reinvestment > MIN_REINVESTMENT_SHARE_OF_REVENUE * total_revenue
        else None
    )
    implied = reinvestment_rate * base_roic if reinvestment_rate is not None and base_roic is not None else None

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
        implied_fundamental_growth_pct=_pct(implied),
        warnings=warnings,
    )


def _scenario(name: str, drivers: ForecastDrivers, schedule: Dict[str, Any]) -> ForecastScenario:
    return ForecastScenario(
        name=name,
        probability=SCENARIO_PROBABILITIES[name],
        revenue_growth_pct=[round(g * 100.0, 2) for g in drivers.revenue_growth],
        operating_margin_pct=[round(m * 100.0, 2) for m in drivers.operating_margin],
        projected_fcfs=schedule["projected_fcfs"],
        revenue_cagr_pct=schedule["revenue_cagr_pct"],
        cumulative_fcf=schedule["cumulative_fcf"],
    )


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value * 100.0, 2)
