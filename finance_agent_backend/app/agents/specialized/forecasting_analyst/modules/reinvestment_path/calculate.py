"""How much must it reinvest to fund that growth? The CapEx path from the year-1 intensity to the steady state.

Reinvestment consistency: CapEx intensity fades linearly from its year-1 level to a steady state in the final
explicit year, chosen so that final-year net reinvestment (CapEx - D&A + dNWC) equals
terminal growth / steady-state RONIC x NOPAT, the same identity the value-driver terminal value uses. The
explicit period therefore hands the terminal value a cash flow that already funds terminal growth, instead of
holding a peak-cycle (or trough) CapEx intensity forever. When final-year NOPAT is not positive the identity
is meaningless, so CapEx fades to maintenance (= D&A) instead.
"""

from typing import List, Tuple

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import ForecastDrivers, linear_fade
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path.thresholds import (
    CAPEX_FADE_WARNING_RATIO,
    CAPEX_INTENSITY_BOUNDS,
)


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
