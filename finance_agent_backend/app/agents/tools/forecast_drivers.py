"""Forecast drivers: historical growth fade, LLM proposal screening and fade extension.

`ForecastDrivers` holds decimals. `ForecastDriverProposal` is what the LLM submits, in WHOLE percent;
`screen_forecast_drivers` converts it once and clamps it to code-owned bounds.

Margin screen: a margin path within 1 pp of the normalized base margin needs no evidence. Beyond that
noise band, a path without a cited forward-looking MD&A margin/cost chunk must stay in a corridor set by
the audited margin trend:
- expanding trend: may hold or continue up by at most 3 pp in total; a decline below base - 1 pp is an
  unsupported reversal and is clamped;
- contracting trend: the mirror image (continue down by at most 3 pp, no unsupported recovery);
- mixed trend (e.g. a spike then a give-back): may move between the base and the audited average margin
  (mean reversion), +/- 1 pp;
- one audited year: base +/- 1 pp.
With accepted margin evidence, only the 3 pp per-year step limit applies.
"""

import math
from typing import AbstractSet, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.agents.tools.forecast_baseline import MARGIN_FLOOR, ForecastBaseline

DEFAULT_TERMINAL_GROWTH = 0.025
BASE_HORIZON_YEARS = 5
EXTENDED_HORIZON_YEARS = 10
FADE_EXTENSION_THRESHOLD = 0.03
FALLBACK_GROWTH_FLOOR = -0.20
FALLBACK_GROWTH_CAP = 0.40
GROWTH_MIN = -0.30
GROWTH_MAX = 1.00
YEAR1_BAND_MIN = 0.15
YEAR1_BAND_RELATIVE = 0.5
MARGIN_STEP_MAX = 0.03
MARGIN_NOISE_BAND = 0.01
UNSUPPORTED_TREND_ALLOWANCE = 0.03
CAPEX_OVERRIDE_MULTIPLES = (0.5, 2.5)
CAPEX_OVERRIDE_CEILING = 0.30
NWC_OVERRIDE_BOUND = 0.50
DECIMAL_READING_MARGIN = 0.5


class ForecastDrivers(BaseModel):
    """Per-year growth and margin paths plus reinvestment intensities (all decimals).

    `capex_pct_of_revenue` is the year-1 intensity; `build_forecast` fills `capex_path`, which fades it
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
        default_factory=list, description="Applied per-year CapEx intensity (decimals); filled by build_forecast"
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
        **_baseline_intensities(baseline),
    )


def screen_forecast_drivers(
    proposal: ForecastDriverProposal,
    baseline: ForecastBaseline,
    allowed_chunk_ids: Set[str],
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    margin_evidence_chunk_ids: AbstractSet[str] = frozenset(),
) -> DriverScreenResult:
    """Never raises: malformed proposals come back with accepted=False and the errors to fix.

    `margin_evidence_chunk_ids` are the allowed MD&A chunks carrying forward-looking margin/cost language
    (see `mdna_guidance.margin_evidence_chunk_ids`); citing one lifts the trend corridor on the margin path.
    """
    errors: List[str] = []
    for name, series in (
        ("revenue_growth_pct", proposal.revenue_growth_pct),
        ("operating_margin_pct", proposal.operating_margin_pct),
    ):
        if len(series) != BASE_HORIZON_YEARS:
            errors.append(f"{name} needs exactly {BASE_HORIZON_YEARS} values, got {len(series)}.")
        elif not all(math.isfinite(v) for v in series):
            errors.append(f"{name} contains a non-finite value.")
    if not errors:
        growth_anchor = _growth_anchor(baseline)
        # Growth is judged on year 1 alone (strictly): later years legitimately fade far from the anchor.
        checks = (
            ("revenue_growth_pct", proposal.revenue_growth_pct,
             [proposal.revenue_growth_pct[0]], terminal_growth if growth_anchor is None else growth_anchor, 1.0),
            ("operating_margin_pct", proposal.operating_margin_pct,
             proposal.operating_margin_pct, baseline.base_operating_margin, DECIMAL_READING_MARGIN),
        )
        for name, series, probe, reference, margin in checks:
            if all(abs(v) < 1.0 for v in series) and _looks_like_decimals(probe, reference, margin):
                errors.append(f"{name} values look like decimals; pass WHOLE percent (31.5 means 31.5%).")
    if errors:
        return DriverScreenResult(accepted=False, errors=errors)

    adjustments: List[DriverAdjustment] = []
    margin_evidence = (
        proposal.margin_citation_chunk_id is not None
        and proposal.margin_citation_chunk_id in allowed_chunk_ids
        and proposal.margin_citation_chunk_id in margin_evidence_chunk_ids
    )
    growth = _screen_growth([v / 100.0 for v in proposal.revenue_growth_pct], baseline, terminal_growth, adjustments)
    margins = _screen_margins(
        [v / 100.0 for v in proposal.operating_margin_pct], baseline, margin_evidence,
        proposal.margin_citation_chunk_id, adjustments,
    )
    capex = _screen_override(
        "capex_pct_of_revenue", proposal.capex_pct_of_revenue, proposal.capex_citation_chunk_id,
        baseline.capex_pct_of_revenue, _capex_bounds(baseline), allowed_chunk_ids, adjustments,
    )
    nwc = _screen_override(
        "nwc_pct_of_incremental_revenue", proposal.nwc_pct_of_incremental_revenue, proposal.nwc_citation_chunk_id,
        baseline.nwc_pct_of_incremental_revenue, (-NWC_OVERRIDE_BOUND, NWC_OVERRIDE_BOUND),
        allowed_chunk_ids, adjustments,
    )
    intensities = {**_baseline_intensities(baseline), "capex_pct_of_revenue": capex, "nwc_pct_of_incremental_revenue": nwc}
    return DriverScreenResult(
        accepted=True,
        drivers=ForecastDrivers(revenue_growth=growth, operating_margin=margins, **intensities),
        adjustments=adjustments,
        evidence_chunk_ids=[c for c in proposal.evidence_chunk_ids if c in allowed_chunk_ids],
        margin_evidence_accepted=margin_evidence,
    )


def _baseline_intensities(baseline: ForecastBaseline) -> dict:
    return dict(
        tax_rate=baseline.tax_rate,
        capex_pct_of_revenue=baseline.capex_pct_of_revenue,
        da_pct_of_revenue=baseline.da_pct_of_revenue,
        nwc_pct_of_incremental_revenue=baseline.nwc_pct_of_incremental_revenue,
        sbc_pct_of_revenue=baseline.sbc_pct_of_revenue,
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _growth_anchor(baseline: ForecastBaseline) -> Optional[float]:
    return next((g for g in (baseline.latest_revenue_growth, baseline.historical_cagr) if g is not None), None)


def _looks_like_decimals(values: List[float], reference: float, margin: float) -> bool:
    """True when the mean of `values` reads closer to `reference` as a decimal than as a whole percent, by at
    least `margin` (ties, e.g. a breakeven company proposing values around 0%, read as whole percent)."""
    mean = sum(values) / len(values)
    return abs(mean - reference) < margin * abs(mean / 100.0 - reference)


def _screen_growth(
    growth: List[float], baseline: ForecastBaseline, terminal_growth: float, adjustments: List[DriverAdjustment]
) -> List[float]:
    """Year 1 is banded around the latest audited growth, widened toward terminal growth so that
    deceleration after a spike (or recovery after a collapse) is never forced back toward the anchor."""
    anchor = _growth_anchor(baseline)
    out: List[float] = []
    for year, g in enumerate(growth, start=1):
        lo, hi = GROWTH_MIN, GROWTH_MAX
        reason = f"outside the {GROWTH_MIN * 100:.0f}% to {GROWTH_MAX * 100:.0f}% bound"
        if year == 1 and anchor is not None:
            band = max(YEAR1_BAND_MIN, YEAR1_BAND_RELATIVE * abs(anchor))
            lo = max(lo, min(anchor - band, terminal_growth - YEAR1_BAND_MIN))
            hi = min(hi, max(anchor + band, terminal_growth + YEAR1_BAND_MIN))
            reason = (
                f"outside the year-1 band of {lo * 100:.2f}% to {hi * 100:.2f}% around the latest audited growth "
                f"of {anchor * 100:.2f}%"
            )
        applied = round(_clamp(g, lo, hi), 4)
        if applied != round(g, 4):
            adjustments.append(DriverAdjustment(
                driver="revenue_growth", year=year, proposed=round(g, 4), applied=applied,
                action="clamped", reason=reason,
            ))
        out.append(applied)
    return out


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


def _screen_margins(
    margins: List[float],
    baseline: ForecastBaseline,
    evidence_accepted: bool,
    citation: Optional[str],
    adjustments: List[DriverAdjustment],
) -> List[float]:
    """3 pp per-year step limit always; without accepted margin evidence, also the trend corridor."""
    corr_lo, corr_hi, corridor = unsupported_margin_corridor(baseline)
    if evidence_accepted:
        corr_lo, corr_hi = -math.inf, math.inf
    evidence_note = (
        f"citation '{citation}' is not an MD&A chunk with forward-looking margin/cost language"
        if citation else "no margin_citation_chunk_id was given"
    )
    prev = baseline.base_operating_margin
    out: List[float] = []
    for year, m in enumerate(margins, start=1):
        step_lo, step_hi = prev - MARGIN_STEP_MAX, prev + MARGIN_STEP_MAX
        lo, hi = max(step_lo, corr_lo, MARGIN_FLOOR), min(step_hi, corr_hi, 1.0)
        if lo > hi:  # rounding drift only: prev always sits inside the corridor
            lo = hi = _clamp(prev, corr_lo, corr_hi)
        applied = round(_clamp(m, lo, hi), 4)
        if applied != round(m, 4):
            step_bound = (m < lo and lo == step_lo) or (m > hi and hi == step_hi)
            reason = (
                f"moves more than {MARGIN_STEP_MAX * 100:.0f} pp from the prior year's {prev * 100:.2f}%"
                if step_bound else f"leaves {corridor}; {evidence_note}"
            )
            adjustments.append(DriverAdjustment(
                driver="operating_margin", year=year, proposed=round(m, 4), applied=applied, action="clamped",
                reason=reason,
            ))
        out.append(applied)
        prev = applied
    return out


def _capex_bounds(baseline: ForecastBaseline) -> Tuple[float, float]:
    if baseline.capex_pct_of_revenue > 0:
        lo_mult, hi_mult = CAPEX_OVERRIDE_MULTIPLES
        return baseline.capex_pct_of_revenue * lo_mult, min(baseline.capex_pct_of_revenue * hi_mult, 1.0)
    return 0.0, CAPEX_OVERRIDE_CEILING


def _screen_override(
    name: str,
    proposed_pct: Optional[float],
    chunk_id: Optional[str],
    baseline_value: float,
    bounds: Tuple[float, float],
    allowed_chunk_ids: Set[str],
    adjustments: List[DriverAdjustment],
) -> float:
    if proposed_pct is None:
        return baseline_value
    proposed = proposed_pct / 100.0
    if not math.isfinite(proposed) or chunk_id not in allowed_chunk_ids:
        adjustments.append(DriverAdjustment(
            driver=name, proposed=proposed if math.isfinite(proposed) else 0.0, applied=baseline_value,
            action="rejected", reason="override needs a finite value and a citation chunk_id from the MD&A excerpts",
        ))
        return baseline_value
    applied = round(_clamp(proposed, *bounds), 4)
    if applied != round(proposed, 4):
        adjustments.append(DriverAdjustment(
            driver=name, proposed=round(proposed, 4), applied=applied, action="clamped",
            reason=f"outside the allowed range {bounds[0] * 100:.2f}% to {bounds[1] * 100:.2f}%",
        ))
    return applied
