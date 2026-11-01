"""Forecast assembly: fade extension, reinvestment path, base schedule, bull/base/bear scenarios and diagnostics.

Reinvestment consistency: CapEx intensity fades linearly from its year-1 level to a steady state in the final
explicit year, chosen so that final-year net reinvestment (CapEx - D&A + dNWC) equals
terminal growth / steady-state RONIC x NOPAT, the same identity the value-driver terminal value uses. The
explicit period therefore hands the terminal value a cash flow that already funds terminal growth, instead of
holding a peak-cycle (or trough) CapEx intensity forever. When final-year NOPAT is not positive the identity
is meaningless, so CapEx fades to maintenance (= D&A) instead.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import ForecastDrivers, extend_fade, linear_fade
from app.agents.tools.forecast_tools import calculate_forecast_schedule, project_forecast_years

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
CAPEX_INTENSITY_BOUNDS = (0.0, 1.0)
CAPEX_FADE_WARNING_RATIO = 2.0


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


def apply_reinvestment_path(
    baseline: ForecastBaseline, drivers: ForecastDrivers, terminal_growth: float
) -> Tuple[ForecastDrivers, List[str]]:
    """Returns drivers with `capex_path` faded from year-1 CapEx to the terminal-consistent steady state.

    Steady state, as decimals of final-year revenue R_N:
        net reinvestment / R_N = g / RONIC_ss x NOPAT margin_N
        CapEx_ss = D&A + net reinvestment - dNWC_N / R_N,  with dNWC_N / R_N = nwc x g_N / (1 + g_N)
    clamped to 0%-100% of revenue. Never raises.
    """
    warnings: List[str] = []
    n = len(drivers.revenue_growth)
    g_final = drivers.revenue_growth[-1]
    nopat_margin = drivers.operating_margin[-1] * (1.0 - drivers.tax_rate)
    nwc_share = drivers.nwc_pct_of_incremental_revenue * g_final / (1.0 + g_final) if g_final > -1.0 else 0.0
    if nopat_margin > 0:
        method = "value_driver_fade"
        target = drivers.da_pct_of_revenue + terminal_growth / baseline.steady_state_ronic * nopat_margin - nwc_share
    else:
        method = "maintenance_fade"
        target = drivers.da_pct_of_revenue
        warnings.append(
            "Final-year NOPAT is not positive, so CapEx fades to maintenance (equal to D&A) rather than to a "
            "growth-funding steady state."
        )
    lo, hi = CAPEX_INTENSITY_BOUNDS
    steady = round(min(max(target, lo), hi), 4)
    if steady != round(target, 4):
        warnings.append(
            f"Steady-state CapEx of {target * 100:.2f}% of revenue is outside {lo * 100:.0f}%-{hi * 100:.0f}%; "
            f"set to {steady * 100:.2f}%."
        )
    start = drivers.capex_pct_of_revenue
    if start > 0 and steady > 0 and max(start, steady) / min(start, steady) >= CAPEX_FADE_WARNING_RATIO:
        warnings.append(
            f"CapEx intensity fades from {start * 100:.2f}% to a {steady * 100:.2f}% steady state; the explicit "
            "years carry a large reinvestment transition."
        )
    path = linear_fade(start, steady, n) if n > 1 else [steady]
    return drivers.model_copy(update={
        "capex_path": path, "steady_state_capex_pct_of_revenue": steady, "reinvestment_method": method,
    }), warnings


def run_schedule(baseline: ForecastBaseline, drivers: ForecastDrivers) -> Dict[str, Any]:
    """Runs the schedule engine; `drivers.capex_path` must already be set (see apply_reinvestment_path)."""
    return calculate_forecast_schedule(
        base_revenue=baseline.base_revenue,
        base_year=baseline.base_year,
        revenue_growth_rates=drivers.revenue_growth,
        operating_margins=drivers.operating_margin,
        tax_rate=drivers.tax_rate,
        capex_pct_of_revenue=drivers.capex_path,
        depreciation_pct_of_revenue=drivers.da_pct_of_revenue,
        nwc_change_pct_of_revenue=drivers.nwc_pct_of_incremental_revenue,
        sbc_pct_of_revenue=drivers.sbc_pct_of_revenue,
        historical_rows=baseline.historical_rows,
    )


def project_with_overrides(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    terminal_growth: float,
    *,
    growth: Optional[List[float]] = None,
    margins: Optional[List[float]] = None,
    terminal_tax: Optional[float] = None,
) -> Tuple[List[float], float]:
    """(UFCFs, final-year NOPAT) for `drivers` with the growth and/or margin path replaced and the CapEx steady
    state rebuilt for the new path. Used by the reverse DCF. With `terminal_tax`, a profitable final year's NOPAT
    is re-taxed at that (marginal) rate, as the headline terminal value does. Raises ValueError on inputs the
    engine rejects."""
    update: Dict[str, Any] = {}
    if growth is not None:
        update["revenue_growth"] = list(growth)
    if margins is not None:
        update["operating_margin"] = list(margins)
    variant, _ = apply_reinvestment_path(baseline, drivers.model_copy(update=update), terminal_growth)
    years = project_forecast_years(
        baseline.base_revenue, baseline.base_year, variant.revenue_growth, variant.operating_margin,
        variant.tax_rate, variant.capex_path, variant.da_pct_of_revenue, variant.nwc_pct_of_incremental_revenue,
        variant.sbc_pct_of_revenue,
    )
    final = years[-1]
    nopat = final.projected_ebit * (1.0 - terminal_tax) if terminal_tax is not None and final.projected_ebit > 0 else final.projected_nopat
    return [y.projected_unlevered_fcf for y in years], nopat


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


def build_scenarios(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    base_schedule: Dict[str, Any],
    risk_profile: Optional[str],
    terminal_growth: float,
) -> List[ForecastScenario]:
    """Growth shifts are largest in year 1 and vanish by the final year, so every case meets terminal growth.

    Bull and bear are defined by free cash flow: when the base case burns cash, faster growth burns more,
    so the growth shift flips sign while the margin shift keeps its direction. Each case rebuilds its own
    CapEx steady state from its final-year NOPAT, matching how the valuation sizes its terminal value.
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
        variant, _ = apply_reinvestment_path(baseline, drivers.model_copy(update={
            "revenue_growth": [round(g + g_shift * w, 4) for g, w in zip(drivers.revenue_growth, weights)],
            "operating_margin": [round(min(max(m + m_shift, -2.0), 1.0), 4) for m in drivers.operating_margin],
        }), terminal_growth)
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
