"""Forecast table: audited history ('A') columns followed by projected ('E') columns."""

from typing import List, Optional

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import HistoricalRow
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule.schema import ForecastYearResult


def _money(value: Optional[float]) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _percent(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def render_forecast_table(
    base_revenue: float,
    base_year: int,
    schedule: List[ForecastYearResult],
    historical_rows: Optional[List[HistoricalRow]],
    tax_rate: float,
) -> str:
    """Markdown table: audited history ('A') columns followed by projected ('E') columns."""
    history = sorted(
        (r for r in historical_rows or [] if r.fiscal_year <= base_year), key=lambda r: r.fiscal_year
    ) or [HistoricalRow(fiscal_year=base_year, revenue=base_revenue)]
    blanks = ["—"] * len(history)
    headers = [f"FY{r.fiscal_year}A" for r in history] + [f"FY{s.projected_year}E" for s in schedule]
    rows = [
        "| Metric ($ Millions) | " + " | ".join(headers) + " |",
        "|:---|" + "|".join([":---:"] * len(headers)) + "|",
    ]

    def add(label: str, hist_cells: List[str], proj_cells: List[str]) -> None:
        rows.append(f"| {label} | " + " | ".join(hist_cells + proj_cells) + " |")

    add("Revenue", [_money(r.revenue) for r in history], [_money(s.projected_revenue) for s in schedule])
    add("Revenue Growth YoY", [_percent(r.revenue_growth_pct, signed=True) for r in history],
        [_percent(s.projected_revenue_growth_pct, signed=True) for s in schedule])
    add("Operating Income (EBIT)", blanks, [_money(s.projected_ebit) for s in schedule])
    add("Operating Margin", [_percent(r.operating_margin_pct) for r in history],
        [_percent(s.projected_ebit_margin_pct) for s in schedule])
    add(f"NOPAT (Effective Tax {tax_rate * 100:.1f}%)", blanks, [_money(s.projected_nopat) for s in schedule])
    add("Capital Expenditures", blanks, [_money(s.projected_capex) for s in schedule])
    add("Depreciation & Amortization", blanks, [_money(s.projected_depreciation) for s in schedule])
    if any(s.projected_nwc_change != 0.0 for s in schedule):
        add("Change in Working Capital (ΔNWC)", blanks, [_money(s.projected_nwc_change) for s in schedule])
    if any(s.projected_sbc != 0.0 for s in schedule):
        add("Stock-Based Compensation (memo, in EBIT)", blanks, [_money(s.projected_sbc) for s in schedule])
    add("**Unlevered Free Cash Flow (UFCF)**", blanks,
        [f"**{_money(s.projected_unlevered_fcf)}**" for s in schedule])
    return "\n".join(rows)
