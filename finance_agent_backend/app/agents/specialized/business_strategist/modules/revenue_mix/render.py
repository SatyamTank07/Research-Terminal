"""Markdown tables of the verified latest-year segment and revenue mix, embedded verbatim in the report."""

from typing import Any, Optional, Sequence

from app.agents.specialized.business_strategist.modules.revenue_mix.schema import RevenueLineFigure, SegmentFigure


def _money(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"({abs(value):,.0f})" if value < 0 else f"{value:,.0f}"


def _pct_cell(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _cell(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def render_segment_table_markdown(segments: Sequence[SegmentFigure]) -> str:
    """Latest-year segment table (revenue, share, profit, margin, profit share, YoY) for verbatim embedding."""
    if not segments:
        return "_No reportable-segment financials were extracted from the filing._"
    latest = max(f.fiscal_year for f in segments)
    rows = [f for f in segments if f.fiscal_year == latest]
    measures = {f.profit_measure for f in rows if f.profit_measure}
    measure = measures.pop() if len(measures) == 1 else "Segment profit"
    lines = [
        f"| Segment | FY{latest} revenue ($M) | % of revenue | YoY growth | {_cell(measure)} ($M) | Margin | % of segment profit |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in rows:
        name = f.segment_name if f.reportable else f"{f.segment_name} (not a reportable segment)"
        lines.append(
            f"| {_cell(name)} | {_money(f.revenue)} | {_pct_cell(f.revenue_share_pct)} | {_pct_cell(f.revenue_growth_pct)} | "
            f"{_money(f.segment_profit)} | {_pct_cell(f.profit_margin_pct)} | {_pct_cell(f.profit_share_pct)} |"
        )
    lines.append(f"\n_Source: 10-K segment note (chunk `{rows[0].source_chunk_id}`); figures in USD millions as reported._")
    return "\n".join(lines)


def render_revenue_mix_markdown(lines_: Sequence[RevenueLineFigure]) -> str:
    """Latest-year product / service revenue mix with sub-lines indented, for verbatim embedding."""
    if not lines_:
        return "_No product / service revenue disaggregation was extracted from the filing._"
    latest = max(f.fiscal_year for f in lines_)
    rows = [f for f in lines_ if f.fiscal_year == latest]
    out = [
        f"| Revenue line | FY{latest} revenue ($M) | % of revenue | YoY growth |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for f in rows:
        name = f"&nbsp;&nbsp;↳ {f.name}" if f.parent else f.name
        out.append(f"| {_cell(name)} | {_money(f.revenue)} | {_pct_cell(f.revenue_share_pct)} | {_pct_cell(f.revenue_growth_pct)} |")
    out.append(f"\n_Source: 10-K revenue disaggregation (chunk `{rows[0].source_chunk_id}`); figures in USD millions as reported._")
    return "\n".join(out)
