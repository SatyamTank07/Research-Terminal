"""How profitable is each dollar of revenue, and is that changing? Opex leverage and margin moves."""

from typing import List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.findings import ForensicFinding, add_finding
from app.agents.specialized.financial_auditor.modules.financial_history import (
    YearFinancialsResult,
    growth_pct,
    operating_expenses_of,
)
from app.agents.specialized.financial_auditor.modules.margins_and_costs.schema import YearCostStructure
from app.agents.specialized.financial_auditor.modules.margins_and_costs.thresholds import (
    MARGIN_CUMULATIVE_DROP_BPS,
    MARGIN_SINGLE_YEAR_DROP_BPS,
    OPEX_VS_REVENUE_GROWTH_THRESHOLD_PP,
)


def measure_cost_structure(
    annual_financials: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
) -> List[YearCostStructure]:
    """Opex growth against revenue growth, and gross / operating margin moves in bps, per year."""
    hist_map = {h.fiscal_year: h for h in history}
    rows: List[YearCostStructure] = []
    prev: Optional[AnnualFinancialInput] = None
    for f in sorted(annual_financials, key=lambda x: x.fiscal_year):
        h = hist_map[f.fiscal_year]
        opex = operating_expenses_of(f)
        opex_growth = rev_growth = gm_bps = om_bps = None
        if prev is not None:
            prev_h = hist_map[prev.fiscal_year]
            opex_growth = growth_pct(opex, operating_expenses_of(prev))
            rev_growth = h.revenue_growth_pct
            if h.gross_margin_pct is not None and prev_h.gross_margin_pct is not None:
                gm_bps = round((h.gross_margin_pct - prev_h.gross_margin_pct) * 100.0, 0)
            om_bps = round((h.operating_margin_pct - prev_h.operating_margin_pct) * 100.0, 0)
        rows.append(YearCostStructure(
            fiscal_year=f.fiscal_year,
            operating_expenses=opex,
            operating_expense_growth_pct=opex_growth,
            revenue_growth_pct=rev_growth,
            opex_growth_minus_revenue_growth_pp=round(opex_growth - rev_growth, 2)
            if opex_growth is not None and rev_growth is not None else None,
            gross_margin_change_bps=gm_bps,
            operating_margin_change_bps=om_bps,
        ))
        prev = f
    return rows


def screen_margins_and_costs(
    history: List[YearFinancialsResult],
    cost_structure: List[YearCostStructure],
) -> List[ForensicFinding]:
    """NEGATIVE_OPERATING_LEVERAGE (opex outgrowing revenue) and MARGIN_COMPRESSION (operating margin falling)."""
    findings: List[ForensicFinding] = []
    cs = {r.fiscal_year: r for r in cost_structure}
    for yr, c in cs.items():
        if c.opex_growth_minus_revenue_growth_pp is not None and \
                c.opex_growth_minus_revenue_growth_pp > OPEX_VS_REVENUE_GROWTH_THRESHOLD_PP:
            add_finding(findings, "NEGATIVE_OPERATING_LEVERAGE", "medium", yr, "opex_growth_minus_revenue_growth_pp",
                c.opex_growth_minus_revenue_growth_pp, OPEX_VS_REVENUE_GROWTH_THRESHOLD_PP,
                f"NEGATIVE OPERATING LEVERAGE (FY{yr}): Operating expenses grew {c.operating_expense_growth_pct:+.2f}% "
                f"against revenue growth of {c.revenue_growth_pct:+.2f}% "
                f"({c.opex_growth_minus_revenue_growth_pp:.1f}pp gap).")

    margin_moves = [(yr, c.operating_margin_change_bps) for yr, c in sorted(cs.items())
                    if c.operating_margin_change_bps is not None]
    if margin_moves:
        worst_yr, worst = min(margin_moves, key=lambda m: m[1])
        cumulative = history[-1].operating_margin_pct - history[0].operating_margin_pct
        all_down = all(m[1] < 0 for m in margin_moves)
        if worst <= -MARGIN_SINGLE_YEAR_DROP_BPS or (all_down and cumulative * 100.0 <= -MARGIN_CUMULATIVE_DROP_BPS):
            add_finding(findings, "MARGIN_COMPRESSION", "medium", history[-1].fiscal_year, "operating_margin_change_bps",
                round(cumulative * 100.0, 0), -MARGIN_CUMULATIVE_DROP_BPS,
                f"MARGIN COMPRESSION: Operating margin moved from {history[0].operating_margin_pct:.2f}% "
                f"(FY{history[0].fiscal_year}) to {history[-1].operating_margin_pct:.2f}% "
                f"(FY{history[-1].fiscal_year}), {cumulative * 100.0:,.0f} bps; worst single year FY{worst_yr} "
                f"({worst:,.0f} bps).")
    return findings
