"""Does the forecast hang together? Reinvestment rate, implied ROIC / RONIC and growth-support cross-checks."""

from typing import Any, Dict, Optional

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import ForecastDrivers
from app.agents.specialized.forecasting_analyst.modules.forecast_consistency.schema import ForecastDiagnostics
from app.agents.specialized.forecasting_analyst.modules.forecast_consistency.thresholds import (
    GROWTH_SUPPORT_GAP_WARNING,
    MIN_REINVESTMENT_SHARE_OF_REVENUE,
    RONIC_WARNING,
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
