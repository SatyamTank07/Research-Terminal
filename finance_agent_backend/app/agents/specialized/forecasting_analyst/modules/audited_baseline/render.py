"""Audited history table shown to the LLM in the forecast task."""

from typing import List

from app.agents.specialized.forecasting_analyst.modules.audited_baseline.schema import HistoricalRow


def render_history_table(rows: List[HistoricalRow]) -> str:
    lines = ["| Fiscal year | Revenue ($M) | Growth | Operating margin |", "|:---|---:|---:|---:|"]
    for r in rows:
        growth = "—" if r.revenue_growth_pct is None else f"{r.revenue_growth_pct:+.2f}%"
        margin = "—" if r.operating_margin_pct is None else f"{r.operating_margin_pct:.2f}%"
        lines.append(f"| FY{r.fiscal_year} | {r.revenue:,.0f} | {growth} | {margin} |")
    return "\n".join(lines)
