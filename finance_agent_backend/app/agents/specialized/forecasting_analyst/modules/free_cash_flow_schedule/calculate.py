"""What cash does the business throw off each year? The deterministic UFCF schedule engine.

Pure-Python projection of revenue, EBIT, NOPAT and unlevered free cash flow:

    UFCF = NOPAT + D&A - CapEx - dNWC

SBC is reported as a memo line only: GAAP operating income already expenses it, so UFCF built from
EBIT treats it as a cash cost without a second deduction.

Every ratio input is a DECIMAL (0.08 = 8%). Callers convert units before calling; the engine never
guesses whether a value is a percent.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline, HistoricalRow
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path import apply_reinvestment_path
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import ForecastDrivers
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule.render import render_forecast_table
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule.schema import (
    ForecastScheduleResult,
    ForecastYearResult,
)
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule.thresholds import MAX_HORIZON_YEARS

logger = logging.getLogger("finance_agent.agents.forecasting_analyst")

Ratio = Union[float, List[float]]


def _finite(val: Any, name: str) -> float:
    if val is None:
        raise ValueError(f"'{name}' is required and cannot be None.")
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be numeric, got: {val!r}")
    if math.isnan(f_val) or math.isinf(f_val):
        raise ValueError(f"'{name}' must be finite, got: {f_val}")
    return f_val


def _bounded(val: Any, name: str, lo: float, hi: float) -> float:
    f_val = _finite(val, name)
    if not lo <= f_val <= hi:
        raise ValueError(f"'{name}' must be a decimal between {lo} and {hi}, got: {f_val}")
    return f_val


def _series(vals: Ratio, name: str, horizon: int, lo: float, hi: float) -> List[float]:
    if isinstance(vals, (list, tuple)):
        if len(vals) != horizon:
            raise ValueError(f"'{name}' must contain exactly {horizon} entries (got {len(vals)}).")
        return [_bounded(v, f"{name}[{i}]", lo, hi) for i, v in enumerate(vals)]
    return [_bounded(vals, name, lo, hi)] * horizon


def project_forecast_years(
    base_revenue: float,
    base_year: int,
    revenue_growth_rates: List[float],
    operating_margins: List[float],
    tax_rate: float,
    capex_pct_of_revenue: Ratio,
    depreciation_pct_of_revenue: Ratio,
    nwc_change_pct_of_revenue: Ratio = 0.0,
    sbc_pct_of_revenue: Ratio = 0.0,
) -> List[ForecastYearResult]:
    """Validated per-year projection without the table or logging (used directly by the reverse DCF).

    All ratios are decimals; dNWC is a share of incremental revenue. Raises ValueError on invalid inputs.
    """
    base_rev = _finite(base_revenue, "base_revenue")
    if base_rev <= 0.0:
        raise ValueError(f"'base_revenue' must be strictly positive, got: {base_rev}")
    if not isinstance(base_year, int) or not 1900 <= base_year <= 2100:
        raise ValueError(f"'base_year' must be a 4-digit year, got: {base_year}")
    if not isinstance(revenue_growth_rates, (list, tuple)) or not 1 <= len(revenue_growth_rates) <= MAX_HORIZON_YEARS:
        raise ValueError(f"'revenue_growth_rates' must be a list of 1 to {MAX_HORIZON_YEARS} decimals.")
    if not isinstance(operating_margins, (list, tuple)):
        raise ValueError("'operating_margins' must be a list of decimals.")

    horizon = len(revenue_growth_rates)
    growth = _series(list(revenue_growth_rates), "revenue_growth_rates", horizon, -0.95, 5.0)
    margins = _series(list(operating_margins), "operating_margins", horizon, -2.0, 1.0)
    tax = _bounded(tax_rate, "tax_rate", 0.0, 0.5)
    capex = _series(capex_pct_of_revenue, "capex_pct_of_revenue", horizon, 0.0, 1.0)
    depr = _series(depreciation_pct_of_revenue, "depreciation_pct_of_revenue", horizon, 0.0, 0.5)
    nwc = _series(nwc_change_pct_of_revenue, "nwc_change_pct_of_revenue", horizon, -0.5, 0.5)
    sbc = _series(sbc_pct_of_revenue, "sbc_pct_of_revenue", horizon, 0.0, 0.5)

    schedule: List[ForecastYearResult] = []
    prev_rev = base_rev
    for i in range(horizon):
        rev = prev_rev * (1.0 + growth[i])
        ebit = rev * margins[i]
        nopat = ebit * (1.0 - tax)
        year_capex = rev * capex[i]
        year_depr = rev * depr[i]
        year_nwc = (rev - prev_rev) * nwc[i]
        year_sbc = rev * sbc[i]
        ufcf = nopat + year_depr - year_capex - year_nwc
        schedule.append(ForecastYearResult(
            projected_year=base_year + i + 1,
            projected_revenue=round(rev, 2),
            projected_revenue_growth_pct=round(growth[i] * 100.0, 2),
            projected_ebit=round(ebit, 2),
            projected_ebit_margin_pct=round(margins[i] * 100.0, 2),
            projected_nopat=round(nopat, 2),
            projected_capex=round(year_capex, 2),
            projected_depreciation=round(year_depr, 2),
            projected_nwc_change=round(year_nwc, 2),
            projected_sbc=round(year_sbc, 2),
            projected_unlevered_fcf=round(ufcf, 2),
        ))
        prev_rev = rev
    return schedule


def calculate_forecast_schedule(
    base_revenue: float,
    base_year: int,
    revenue_growth_rates: List[float],
    operating_margins: List[float],
    tax_rate: float,
    capex_pct_of_revenue: Ratio,
    depreciation_pct_of_revenue: Ratio,
    nwc_change_pct_of_revenue: Ratio = 0.0,
    sbc_pct_of_revenue: Ratio = 0.0,
    historical_rows: Optional[List[HistoricalRow]] = None,
) -> Dict[str, Any]:
    """Projects the explicit-horizon schedule. All ratios are decimals; dNWC is a share of incremental revenue."""
    schedule = project_forecast_years(
        base_revenue, base_year, revenue_growth_rates, operating_margins, tax_rate, capex_pct_of_revenue,
        depreciation_pct_of_revenue, nwc_change_pct_of_revenue, sbc_pct_of_revenue,
    )
    base_rev = float(base_revenue)
    tax = float(tax_rate)
    horizon = len(schedule)
    final_rev = base_rev
    for g in revenue_growth_rates:
        final_rev *= 1.0 + float(g)
    projected_fcfs = [s.projected_unlevered_fcf for s in schedule]
    cumulative = sum(projected_fcfs)
    result = ForecastScheduleResult(
        base_revenue=round(base_rev, 2),
        base_year=base_year,
        forecast_horizon_years=horizon,
        revenue_cagr_pct=round((math.pow(final_rev / base_rev, 1.0 / horizon) - 1.0) * 100.0, 2),
        cumulative_fcf=round(cumulative, 2),
        average_annual_fcf=round(cumulative / horizon, 2),
        tax_rate_pct=round(tax * 100.0, 2),
        projected_fcfs=projected_fcfs,
        forecast_schedule=schedule,
        forecast_table_markdown=render_forecast_table(base_rev, base_year, schedule, historical_rows, tax),
    )
    logger.info(
        f"calculate_forecast_schedule: base ${base_rev:,.0f}M FY{base_year}, {horizon}y, "
        f"CAGR {result.revenue_cagr_pct:.2f}%, cumulative UFCF ${result.cumulative_fcf:,.0f}M"
    )
    return result.model_dump()


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
