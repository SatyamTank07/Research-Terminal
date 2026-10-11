"""Deterministic forecast baseline derived from the Financial Auditor's output.

Every auditor field ending in `_pct` is a whole percent; this module converts each one to a decimal
exactly once, so the forecast engine never guesses units.

Code-owned assumptions (chosen to hold for any company, not tuned to one ticker):

- Starting CapEx intensity is a recency-weighted average of up to three audited years (weights 1:2:3,
  latest heaviest, so the latest year carries half the weight; two years use 2:3). An investment step-up or cut usually
  carries into the next year (committed projects, construction in progress), so a flat mean understates
  it, but one lumpy year should not set the whole path. The forecast then fades CapEx to a steady state.
- Steady-state return on new invested capital (RONIC) = base-year ROIC (NOPAT / (total debt + equity)),
  clamped to [8%, 30%]. Below 8% a company would reinvest forever at returns under almost any cost of
  capital; above 30%, perpetual returns on *incremental* capital are rare because competition erodes
  them. Without a positive ROIC (losses, negative capital) it defaults to 10%, roughly the long-run
  median ROIC of US non-financial companies. The forecaster does not know WACC; the valuation separately
  floors the terminal RONIC at WACC.
- Margin trend: classified from the last three audited operating margins (expanding / contracting /
  mixed) so the LLM driver screen can tell trend-consistent paths from unsupported reversals.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.agents.tools.financial_math_tools import STATUTORY_TAX_RATE
from app.agents.tools.forecast_tools import HistoricalRow

if TYPE_CHECKING:  # runtime import would cycle through app.agents.specialized
    from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput

CAPEX_AVERAGE_YEARS = 3
CAPEX_RECENCY_WEIGHTS = (1.0, 2.0, 3.0)  # oldest -> latest over the last CAPEX_AVERAGE_YEARS audited years
STEADY_STATE_RONIC_FLOOR = 0.08
STEADY_STATE_RONIC_CAP = 0.30
STEADY_STATE_RONIC_DEFAULT = 0.10
MARGIN_TREND_YEARS = 3
MARGIN_TREND_MIN_CHANGE = 0.01
MAX_PLAUSIBLE_TAX_RATE_PCT = 50.0
MARGIN_FLOOR = -2.0
NWC_INTENSITY_BOUND = 0.5


class ForecastBaseline(BaseModel):
    """Audited starting point and reinvestment intensities for the forecast (ratios are decimals)."""

    base_year: int
    base_revenue: float = Field(..., description="Base-year revenue ($M)")
    base_operating_margin: float = Field(..., description="Base-year EBIT margin (decimal)")
    margin_source: Literal["normalized", "reported"]
    latest_revenue_growth: Optional[float] = None
    historical_cagr: Optional[float] = None
    history_years: int
    growth_volatility: Optional[float] = Field(None, description="Population std-dev of yearly revenue growth")
    tax_rate: float
    tax_source: Literal["median_effective", "statutory_default"]
    capex_pct_of_revenue: float = Field(
        ..., description="Starting (year-1) CapEx intensity: recency-weighted (1:2:3) average of the audited years"
    )
    capex_average_pct_of_revenue: float = Field(..., description="Simple average CapEx intensity, same years")
    latest_capex_pct_of_revenue: float = Field(..., description="Base-year CapEx / revenue")
    capex_years: int
    da_pct_of_revenue: float
    da_source: Literal["reported", "ebitda_bridge", "capex_parity_proxy"]
    nwc_pct_of_incremental_revenue: float
    nwc_source: Literal["working_capital_days", "not_available"]
    sbc_pct_of_revenue: float
    sbc_source: Literal["reported", "not_reported"]
    base_invested_capital: Optional[float] = Field(None, description="Total debt + stockholders' equity ($M)")
    base_roic: Optional[float] = Field(
        None, description="Base-year NOPAT / invested capital (decimal); None when invested capital is not positive"
    )
    steady_state_ronic: float = Field(
        ..., description="Return on new invested capital that sizes final-year net reinvestment (decimal)"
    )
    steady_state_ronic_source: Literal[
        "base_roic", "base_roic_floored", "base_roic_capped", "default_no_positive_roic"
    ]
    margin_trend: Literal["expanding", "contracting", "mixed", "insufficient_history"] = Field(
        ..., description="Direction of the last three audited (reported) operating margins"
    )
    historical_average_margin: Optional[float] = Field(
        None, description="Mean audited operating margin over the history (decimal); None with one year"
    )
    historical_rows: List[HistoricalRow] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def build_forecast_baseline(audit: FinancialAuditOutput) -> ForecastBaseline:
    history = sorted(audit.multi_year_history, key=lambda h: h.fiscal_year)
    if not history:
        raise ValueError("Forecast baseline needs at least one audited year.")
    latest = history[-1]
    if latest.revenue <= 0:
        raise ValueError(f"Forecast baseline needs positive base-year revenue, got {latest.revenue}.")

    warnings: List[str] = []
    eq_by_year = {e.fiscal_year: e for e in audit.earnings_quality}
    wc_by_year = {w.fiscal_year: w for w in audit.working_capital}
    latest_eq = eq_by_year.get(latest.fiscal_year)

    margin, margin_source = _base_margin(latest, latest_eq)
    tax_rate, tax_source = _tax_rate(history, eq_by_year)
    capex_window = [h for h in history[-CAPEX_AVERAGE_YEARS:] if h.revenue > 0]
    capex_ratios = [h.capital_expenditures / h.revenue for h in capex_window]
    capex_avg = statistics.fmean(capex_ratios)
    weights = CAPEX_RECENCY_WEIGHTS[-len(capex_ratios):]
    capex_start = sum(w * r for w, r in zip(weights, capex_ratios)) / sum(weights)
    da_pct, da_source = _da_intensity(latest, audit, capex_avg)
    if da_source == "capex_parity_proxy":
        warnings.append("D&A not reported; assumed equal to CapEx intensity (maintenance-level reinvestment).")
    nwc_pct, nwc_source = _nwc_intensity(wc_by_year.get(latest.fiscal_year), latest, warnings)
    sbc = latest_eq.stock_based_compensation if latest_eq is not None else None
    growths = [h.revenue_growth_pct / 100.0 for h in history if h.revenue_growth_pct is not None]
    invested = audit.balance_sheet.total_debt + audit.balance_sheet.stockholders_equity
    base_margin = _capped(margin, MARGIN_FLOOR, 1.0, "Base operating margin", warnings)
    base_roic = latest.revenue * base_margin * (1.0 - tax_rate) / invested if invested > 0 else None
    ronic_ss, ronic_source = _steady_state_ronic(base_roic)
    if ronic_source == "default_no_positive_roic":
        warnings.append(
            f"Base-year ROIC is not positive; steady-state reinvestment assumes a "
            f"{STEADY_STATE_RONIC_DEFAULT * 100:.0f}% return on new capital."
        )
    reported_margins = [h.operating_margin_pct / 100.0 for h in history]

    return ForecastBaseline(
        base_year=latest.fiscal_year,
        base_revenue=latest.revenue,
        base_operating_margin=base_margin,
        margin_source=margin_source,
        latest_revenue_growth=None if latest.revenue_growth_pct is None else latest.revenue_growth_pct / 100.0,
        historical_cagr=_cagr(history),
        history_years=len(history),
        growth_volatility=statistics.pstdev(growths) if len(growths) >= 2 else None,
        tax_rate=tax_rate,
        tax_source=tax_source,
        capex_pct_of_revenue=_capped(capex_start, 0.0, 1.0, "CapEx intensity", warnings),
        capex_average_pct_of_revenue=min(max(capex_avg, 0.0), 1.0),
        latest_capex_pct_of_revenue=min(max(capex_ratios[-1], 0.0), 1.0),
        capex_years=len(capex_window),
        da_pct_of_revenue=_capped(da_pct, 0.0, 0.5, "D&A intensity", warnings),
        da_source=da_source,
        nwc_pct_of_incremental_revenue=nwc_pct,
        nwc_source=nwc_source,
        sbc_pct_of_revenue=_capped(sbc / latest.revenue if sbc else 0.0, 0.0, 0.5, "SBC intensity", warnings),
        sbc_source="reported" if sbc is not None else "not_reported",
        base_invested_capital=invested if invested > 0 else None,
        base_roic=base_roic,
        steady_state_ronic=ronic_ss,
        steady_state_ronic_source=ronic_source,
        margin_trend=_margin_trend(reported_margins),
        historical_average_margin=statistics.fmean(reported_margins) if len(reported_margins) >= 2 else None,
        historical_rows=[
            HistoricalRow(
                fiscal_year=h.fiscal_year,
                revenue=h.revenue,
                revenue_growth_pct=h.revenue_growth_pct,
                operating_margin_pct=h.operating_margin_pct,
            )
            for h in history
        ],
        warnings=warnings,
    )


def render_history_table(rows: List[HistoricalRow]) -> str:
    lines = ["| Fiscal year | Revenue ($M) | Growth | Operating margin |", "|:---|---:|---:|---:|"]
    for r in rows:
        growth = "—" if r.revenue_growth_pct is None else f"{r.revenue_growth_pct:+.2f}%"
        margin = "—" if r.operating_margin_pct is None else f"{r.operating_margin_pct:.2f}%"
        lines.append(f"| FY{r.fiscal_year} | {r.revenue:,.0f} | {growth} | {margin} |")
    return "\n".join(lines)


def _base_margin(latest, latest_eq) -> Tuple[float, str]:
    if latest_eq is not None and latest_eq.operating_income_adjustments_total:
        return latest_eq.normalized_operating_margin_pct / 100.0, "normalized"
    return latest.operating_margin_pct / 100.0, "reported"


def _tax_rate(history, eq_by_year: Dict[int, object]) -> Tuple[float, str]:
    """Median effective rate over years with positive pre-tax income; statutory otherwise."""
    rates: List[float] = []
    for h in history:
        e = eq_by_year.get(h.fiscal_year)
        if e is None or e.effective_tax_rate_pct is None or e.non_operating_income is None:
            continue
        pretax = h.operating_income + e.non_operating_income
        if pretax > 0 and 0.0 < e.effective_tax_rate_pct <= MAX_PLAUSIBLE_TAX_RATE_PCT:
            rates.append(e.effective_tax_rate_pct)
    if rates:
        return statistics.median(rates) / 100.0, "median_effective"
    return STATUTORY_TAX_RATE, "statutory_default"


def _da_intensity(latest, audit, capex_pct: float) -> Tuple[float, str]:
    if latest.depreciation_amortization is not None and latest.depreciation_amortization > 0:
        return latest.depreciation_amortization / latest.revenue, "reported"
    ebitda = audit.solvency_and_liquidity_ratios.ebitda
    if ebitda is not None and ebitda - latest.operating_income > 0:
        return (ebitda - latest.operating_income) / latest.revenue, "ebitda_bridge"
    return capex_pct, "capex_parity_proxy"


def _nwc_intensity(wc, latest, warnings: List[str]) -> Tuple[float, str]:
    """Trade working capital / revenue from DSO, DIO and DPO (DIO and DPO are on cost of revenue).

    Needs both DSO and DPO (as the auditor's cash conversion cycle does): one side alone can flip the sign.
    """
    if wc is None or (wc.dso_days is None and wc.dio_days is None and wc.dpo_days is None):
        return 0.0, "not_available"
    if wc.dso_days is None or wc.dpo_days is None:
        warnings.append("Working capital days incomplete (receivable or payable days missing); NWC investment set to 0.")
        return 0.0, "not_available"
    cogs_ratio = 1.0 - latest.gross_margin_pct / 100.0 if latest.gross_margin_pct is not None else 1.0
    days = wc.dso_days + ((wc.dio_days or 0.0) - wc.dpo_days) * cogs_ratio
    return min(max(days / 365.0, -NWC_INTENSITY_BOUND), NWC_INTENSITY_BOUND), "working_capital_days"


def _steady_state_ronic(base_roic: Optional[float]) -> Tuple[float, str]:
    if base_roic is None or base_roic <= 0:
        return STEADY_STATE_RONIC_DEFAULT, "default_no_positive_roic"
    if base_roic < STEADY_STATE_RONIC_FLOOR:
        return STEADY_STATE_RONIC_FLOOR, "base_roic_floored"
    if base_roic > STEADY_STATE_RONIC_CAP:
        return STEADY_STATE_RONIC_CAP, "base_roic_capped"
    return base_roic, "base_roic"


def _margin_trend(margins: List[float]) -> str:
    """Expanding / contracting when the last three audited margins moved at least 1 pp net in one direction
    and the latest year did not reverse it; mixed otherwise (e.g. a spike followed by a give-back)."""
    window = margins[-MARGIN_TREND_YEARS:]
    if len(window) < 2:
        return "insufficient_history"
    net, last = window[-1] - window[0], window[-1] - window[-2]
    if net >= MARGIN_TREND_MIN_CHANGE and last >= 0:
        return "expanding"
    if net <= -MARGIN_TREND_MIN_CHANGE and last <= 0:
        return "contracting"
    return "mixed"


def _cagr(history) -> Optional[float]:
    first, last = history[0], history[-1]
    if len(history) < 2 or first.revenue <= 0:
        return None
    return (last.revenue / first.revenue) ** (1.0 / (len(history) - 1)) - 1.0


def _capped(value: float, lo: float, hi: float, label: str, warnings: List[str]) -> float:
    if lo <= value <= hi:
        return value
    bounded = min(max(value, lo), hi)
    warnings.append(f"{label} of {value * 100:.2f}% is outside {lo * 100:.0f}% to {hi * 100:.0f}%; capped at {bounded * 100:.2f}%.")
    return bounded
