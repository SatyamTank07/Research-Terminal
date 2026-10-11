"""Deterministic forecast baseline derived from the Financial Auditor's output.

Every auditor field ending in `_pct` is a whole percent; this module converts each one to a decimal
exactly once, so the forecast engine never guesses units.
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
    capex_pct_of_revenue: float
    capex_years: int
    da_pct_of_revenue: float
    da_source: Literal["reported", "ebitda_bridge", "capex_parity_proxy"]
    nwc_pct_of_incremental_revenue: float
    nwc_source: Literal["working_capital_days", "not_available"]
    sbc_pct_of_revenue: float
    sbc_source: Literal["reported", "not_reported"]
    base_invested_capital: Optional[float] = Field(None, description="Total debt + stockholders' equity ($M)")
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
    capex_pct = statistics.fmean(h.capital_expenditures / h.revenue for h in capex_window)
    da_pct, da_source = _da_intensity(latest, audit, capex_pct)
    if da_source == "capex_parity_proxy":
        warnings.append("D&A not reported; assumed equal to CapEx intensity (maintenance-level reinvestment).")
    nwc_pct, nwc_source = _nwc_intensity(wc_by_year.get(latest.fiscal_year), latest, warnings)
    sbc = latest_eq.stock_based_compensation if latest_eq is not None else None
    growths = [h.revenue_growth_pct / 100.0 for h in history if h.revenue_growth_pct is not None]
    invested = audit.balance_sheet.total_debt + audit.balance_sheet.stockholders_equity

    return ForecastBaseline(
        base_year=latest.fiscal_year,
        base_revenue=latest.revenue,
        base_operating_margin=_capped(margin, MARGIN_FLOOR, 1.0, "Base operating margin", warnings),
        margin_source=margin_source,
        latest_revenue_growth=None if latest.revenue_growth_pct is None else latest.revenue_growth_pct / 100.0,
        historical_cagr=_cagr(history),
        history_years=len(history),
        growth_volatility=statistics.pstdev(growths) if len(growths) >= 2 else None,
        tax_rate=tax_rate,
        tax_source=tax_source,
        capex_pct_of_revenue=_capped(capex_pct, 0.0, 1.0, "CapEx intensity", warnings),
        capex_years=len(capex_window),
        da_pct_of_revenue=_capped(da_pct, 0.0, 0.5, "D&A intensity", warnings),
        da_source=da_source,
        nwc_pct_of_incremental_revenue=nwc_pct,
        nwc_source=nwc_source,
        sbc_pct_of_revenue=_capped(sbc / latest.revenue if sbc else 0.0, 0.0, 0.5, "SBC intensity", warnings),
        sbc_source="reported" if sbc is not None else "not_reported",
        base_invested_capital=invested if invested > 0 else None,
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
