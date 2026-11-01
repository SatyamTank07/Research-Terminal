"""How fast will revenue grow, and where will margins go? Code-built paths: the historical fade (default and
fallback), the 10-year fade extension, the growth anchor and the margin corridor for unsupported paths.

`ForecastDrivers` holds decimals. The LLM's whole-percent proposal is screened in `verify.py`.
"""

from typing import List, Optional, Tuple

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers.schema import ForecastDrivers
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers.thresholds import (
    BASE_HORIZON_YEARS,
    EXTENDED_HORIZON_YEARS,
    FADE_EXTENSION_THRESHOLD,
    FALLBACK_GROWTH_CAP,
    FALLBACK_GROWTH_FLOOR,
    MARGIN_NOISE_BAND,
    UNSUPPORTED_TREND_ALLOWANCE,
)


def linear_fade(start: float, end: float, years: int) -> List[float]:
    """Year 1 = start, final year = end, linear in between (rounded to 4 dp)."""
    if years < 1:
        raise ValueError("years must be >= 1")
    if years == 1:
        return [round(end, 4)]
    step = (end - start) / (years - 1)
    return [round(start + step * i, 4) for i in range(years)]


def extend_fade(
    growth: List[float], margins: List[float], terminal_growth: float
) -> Tuple[List[float], List[float], bool]:
    """Appends fade years to the 10-year horizon when final growth is far from terminal growth."""
    if len(growth) >= EXTENDED_HORIZON_YEARS or abs(growth[-1] - terminal_growth) <= FADE_EXTENSION_THRESHOLD:
        return list(growth), list(margins), False
    extra = EXTENDED_HORIZON_YEARS - len(growth)
    tail = linear_fade(growth[-1], terminal_growth, extra + 1)[1:]
    return list(growth) + tail, list(margins) + [margins[-1]] * extra, True


def historical_fade_drivers(baseline: ForecastBaseline, terminal_growth: float) -> ForecastDrivers:
    """No-LLM path: growth fades linearly from the historical trend to terminal; margin held flat."""
    trend = next(
        (g for g in (baseline.historical_cagr, baseline.latest_revenue_growth) if g is not None),
        terminal_growth,
    )
    start = min(max(trend, FALLBACK_GROWTH_FLOOR), FALLBACK_GROWTH_CAP)
    years = EXTENDED_HORIZON_YEARS if abs(start - terminal_growth) > FADE_EXTENSION_THRESHOLD else BASE_HORIZON_YEARS
    return ForecastDrivers(
        revenue_growth=linear_fade(start, terminal_growth, years),
        operating_margin=[round(baseline.base_operating_margin, 4)] * years,
        **baseline_intensities(baseline),
    )


def baseline_intensities(baseline: ForecastBaseline) -> dict:
    return dict(
        tax_rate=baseline.tax_rate,
        capex_pct_of_revenue=baseline.capex_pct_of_revenue,
        da_pct_of_revenue=baseline.da_pct_of_revenue,
        nwc_pct_of_incremental_revenue=baseline.nwc_pct_of_incremental_revenue,
        sbc_pct_of_revenue=baseline.sbc_pct_of_revenue,
    )


def growth_anchor(baseline: ForecastBaseline) -> Optional[float]:
    return next((g for g in (baseline.latest_revenue_growth, baseline.historical_cagr) if g is not None), None)


def unsupported_margin_corridor(baseline: ForecastBaseline) -> Tuple[float, float, str]:
    """(low, high, description) bounds for a margin path that cites no MD&A margin/cost evidence."""
    m0 = baseline.base_operating_margin
    tol = MARGIN_NOISE_BAND
    trend = baseline.margin_trend
    if trend == "expanding":
        return m0 - tol, m0 + UNSUPPORTED_TREND_ALLOWANCE, (
            f"the expanding audited margin trend (hold within {tol * 100:.0f} pp of the {m0 * 100:.2f}% base or "
            f"continue up by at most {UNSUPPORTED_TREND_ALLOWANCE * 100:.0f} pp)"
        )
    if trend == "contracting":
        return m0 - UNSUPPORTED_TREND_ALLOWANCE, m0 + tol, (
            f"the contracting audited margin trend (hold within {tol * 100:.0f} pp of the {m0 * 100:.2f}% base or "
            f"continue down by at most {UNSUPPORTED_TREND_ALLOWANCE * 100:.0f} pp)"
        )
    avg = baseline.historical_average_margin
    if trend == "mixed" and avg is not None:
        return min(m0, avg) - tol, max(m0, avg) + tol, (
            f"the mixed audited margin history (between the {m0 * 100:.2f}% base and the {avg * 100:.2f}% "
            f"audited average, +/- {tol * 100:.0f} pp)"
        )
    return m0 - tol, m0 + tol, f"{tol * 100:.0f} pp of the {m0 * 100:.2f}% base margin (one audited year)"
