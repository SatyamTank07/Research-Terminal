"""What is the current price already assuming? Reverse DCF: the constant explicit-period revenue growth, the
uniform operating-margin shift and the discount rate at which the base-case DCF equals the share price, all other
inputs held fixed. The solver scans a grid then bisects, so it is bounded and never raises."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from app.agents.specialized.forecasting_analyst import MARGIN_FLOOR, project_with_overrides
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import CostOfCapital
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.calculate import dcf_fair_value_per_share
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.schema import IntrinsicValue
from app.agents.specialized.valuation_specialist.modules.market_implied_expectations.schema import (
    MarketImpliedExpectations,
)
from app.agents.specialized.valuation_specialist.modules.market_implied_expectations.thresholds import (
    IMPLIED_WACC_MIN_SPREAD_OVER_G,
    IMPLIED_WACC_UPPER,
    REVERSE_DCF_GAP_WARNING_PP,
    REVERSE_DCF_GROWTH_BOUNDS,
    REVERSE_DCF_MARGIN_SHIFT_BOUNDS,
    REVERSE_DCF_MAX_ITERATIONS,
    REVERSE_DCF_SCAN_POINTS,
    REVERSE_DCF_TOLERANCE,
)

if TYPE_CHECKING:
    from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput


def solve_reverse_dcf(
    value_fn: Callable[[float], Optional[float]],
    target: float,
    bounds: Tuple[float, float] = REVERSE_DCF_GROWTH_BOUNDS,
    anchor: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Finds x in `bounds` with value_fn(x) == target. Never raises; at most
    REVERSE_DCF_SCAN_POINTS + REVERSE_DCF_MAX_ITERATIONS evaluations.

    Returns (root, 'solved'), or (None, 'price_above_range' | 'price_below_range' | 'no_solution').
    `value_fn` may return None (no finite value); such points are skipped. With several brackets (a
    non-monotonic value curve) the one nearest `anchor` wins.
    """
    def f(x: float) -> Optional[float]:
        try:
            v = value_fn(x)
        except Exception:  # noqa: BLE001 - the solver must never raise
            return None
        return v - target if v is not None and math.isfinite(v) else None

    lo, hi = bounds
    xs = [lo + (hi - lo) * i / (REVERSE_DCF_SCAN_POINTS - 1) for i in range(REVERSE_DCF_SCAN_POINTS)]
    points = [(x, fx) for x, fx in ((x, f(x)) for x in xs) if fx is not None]
    if not points:
        return None, "no_solution"
    exact = [x for x, fx in points if fx == 0.0]
    brackets = [
        (a, fa, b, fb) for (a, fa), (b, fb) in zip(points, points[1:]) if fa != 0.0 and fb != 0.0 and (fa < 0) != (fb < 0)
    ]
    if not exact and not brackets:
        return None, ("price_above_range" if all(fx < 0 for _, fx in points) else "price_below_range")
    ref = anchor if anchor is not None else (lo + hi) / 2.0
    candidates = [(abs(x - ref), x, None) for x in exact] + [
        (abs((br[0] + br[2]) / 2.0 - ref), None, br) for br in brackets
    ]
    _, exact_root, bracket = min(candidates, key=lambda c: c[0])
    if bracket is None:
        return exact_root, "solved"
    a, fa, b, fb = bracket
    for _ in range(REVERSE_DCF_MAX_ITERATIONS):
        mid = (a + b) / 2.0
        fm = f(mid)
        if fm is None:  # undefined inside the bracket: settle on the bracket midpoint
            break
        if fm == 0.0 or (b - a) / 2.0 < REVERSE_DCF_TOLERANCE:
            return mid, "solved"
        if (fa < 0) != (fm < 0):
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2.0, "solved"


def _reverse_dcf(
    forecast: Optional["ForecastOutput"],
    price: Optional[float],
    terminal_growth_rate: float,
    dcf_kwargs: dict,
    warnings: List[str],
    terminal_tax: float,
) -> dict:
    """Market-implied constant explicit-period revenue growth, plus the uniform operating-margin shift that
    reaches the price at modeled growth; never raises."""
    empty = dict(
        market_implied_growth_pct=None, modeled_revenue_cagr_pct=None, reverse_dcf_horizon_years=None,
        market_implied_margin_shift_pp=None, modeled_final_margin_pct=None,
    )
    if forecast is None or not forecast.applied_drivers.revenue_growth:
        return {**empty, "reverse_dcf_status": "forecast_unavailable",
                "reverse_dcf_summary": "No forecast schedule, so no market-implied growth was solved."}
    horizon = len(forecast.applied_drivers.revenue_growth)
    modeled = forecast.revenue_cagr_pct
    baseline, drivers = forecast.assumptions, forecast.applied_drivers
    final_margin = drivers.operating_margin[-1]
    base = {
        **empty, "modeled_revenue_cagr_pct": modeled, "reverse_dcf_horizon_years": horizon,
        "modeled_final_margin_pct": round(final_margin * 100.0, 2),
    }
    if price is None:
        return {**base, "reverse_dcf_status": "price_unavailable",
                "reverse_dcf_summary": "No share price, so no market-implied growth was solved."}

    def value_at_growth(growth: float) -> Optional[float]:
        fcfs, nopat = project_with_overrides(
            baseline, drivers, terminal_growth_rate, growth=[growth] * horizon, terminal_tax=terminal_tax
        )
        return dcf_fair_value_per_share(fcfs, terminal_nopat=nopat, **dcf_kwargs)

    def value_at_margin_shift(shift: float) -> Optional[float]:
        margins = [min(max(m + shift, MARGIN_FLOOR), 1.0) for m in drivers.operating_margin]
        fcfs, nopat = project_with_overrides(
            baseline, drivers, terminal_growth_rate, margins=margins, terminal_tax=terminal_tax
        )
        return dcf_fair_value_per_share(fcfs, terminal_nopat=nopat, **dcf_kwargs)

    root, status = solve_reverse_dcf(value_at_growth, price, anchor=modeled / 100.0)
    shift, shift_status = solve_reverse_dcf(value_at_margin_shift, price, REVERSE_DCF_MARGIN_SHIFT_BOUNDS, anchor=0.0)
    shift_pp = round(shift * 100.0, 2) if shift_status == "solved" else None
    if shift_pp is not None:
        margin_text = (
            f" At modeled growth the price instead needs a {shift_pp:+.1f} pp operating-margin shift in every year "
            f"(final-year margin {min(max(final_margin + shift, MARGIN_FLOOR), 1.0) * 100:.1f}% versus "
            f"{final_margin * 100:.1f}% modeled)."
        )
    else:
        lo_m, hi_m = REVERSE_DCF_MARGIN_SHIFT_BOUNDS
        margin_text = (
            f" No uniform operating-margin shift between {lo_m * 100:+.0f} and {hi_m * 100:+.0f} pp reaches the price "
            f"at modeled growth."
        )
    base["market_implied_margin_shift_pp"] = shift_pp

    lo, hi = REVERSE_DCF_GROWTH_BOUNDS
    if status == "solved":
        implied = round(root * 100.0, 2)
        gap = implied - modeled
        summary = (
            f"At ${price:.2f} the market implies {implied:.1f}% constant revenue growth for {horizon} years (then "
            f"{terminal_growth_rate * 100:.1f}% terminal), holding modeled margins, reinvestment, WACC and terminal "
            f"RONIC, versus a {modeled:.1f}% modeled revenue CAGR." + margin_text
        )
        if abs(gap) > REVERSE_DCF_GAP_WARNING_PP:
            warnings.append(
                f"Reverse DCF: the share price implies {implied:.1f}% annual revenue growth for {horizon} years versus "
                f"{modeled:.1f}% modeled ({gap:+.1f} pp); the market is pricing materially "
                f"{'more' if gap > 0 else 'less'} growth, or different margins and returns, than the base case."
            )
        return {**base, "market_implied_growth_pct": implied, "reverse_dcf_status": "solved",
                "reverse_dcf_summary": summary}
    if status == "price_above_range":
        summary = (
            f"The ${price:.2f} share price exceeds the DCF value even at {hi * 100:.0f}% constant revenue growth for "
            f"{horizon} years at modeled margins."
        )
    elif status == "price_below_range":
        summary = (
            f"The ${price:.2f} share price is below the DCF value even at {lo * 100:.0f}% constant revenue growth for "
            f"{horizon} years at modeled margins."
        )
    else:
        summary = "The reverse DCF found no finite valuation across the growth range."
    summary += margin_text
    warnings.append(f"Reverse DCF: {summary}")
    return {**base, "reverse_dcf_status": status, "reverse_dcf_summary": summary}


def _implied_discount_rate(
    projected_fcfs: List[float], price: Optional[float], terminal_growth_rate: float, dcf_kwargs: dict,
    wacc: float,
) -> Optional[float]:
    """Discount rate that makes the base-case DCF equal the share price, holding cash flows, terminal RONIC and
    the bridge fixed (the market's required return on these cash flows). None without a price or a root."""
    if price is None:
        return None
    bounds = (terminal_growth_rate + IMPLIED_WACC_MIN_SPREAD_OVER_G, IMPLIED_WACC_UPPER)
    root, status = solve_reverse_dcf(
        lambda w: dcf_fair_value_per_share(projected_fcfs, w, terminal_growth_rate, **dcf_kwargs),
        price, bounds, anchor=wacc,
    )
    return round(root * 100.0, 2) if status == "solved" else None


def solve_market_implied_expectations(
    forecast: Optional["ForecastOutput"], intrinsic: IntrinsicValue, capital: CostOfCapital, price: Optional[float]
) -> MarketImpliedExpectations:
    """Implied growth, implied margin shift and implied discount rate, on the same WACC, terminal value and bridge
    as the headline DCF. A placeholder forecast gives no reverse DCF."""
    warnings: List[str] = []
    g = intrinsic.terms.terminal_growth_rate
    reverse = _reverse_dcf(
        forecast if intrinsic.forecast_source == "forecasting_analyst" else None, price, g,
        intrinsic.terms.as_kwargs(), warnings, capital.marginal_tax_rate,
    )
    implied_wacc = _implied_discount_rate(
        intrinsic.projected_fcfs, price, g,
        {**intrinsic.terms.as_kwargs(without=("wacc", "terminal_growth_rate")), "terminal_nopat": intrinsic.terminal_nopat},
        capital.wacc,
    )
    if implied_wacc is not None:
        reverse["reverse_dcf_summary"] = (
            f"{reverse['reverse_dcf_summary']} Holding the modeled cash flows, the price corresponds to a "
            f"{implied_wacc:.2f}% discount rate versus the {capital.wacc * 100:.2f}% applied."
        ).strip()
    return MarketImpliedExpectations(**reverse, market_implied_wacc_pct=implied_wacc, warnings=tuple(warnings))
