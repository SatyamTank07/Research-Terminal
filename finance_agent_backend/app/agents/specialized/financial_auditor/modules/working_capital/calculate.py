"""Is cash getting tied up in operations? DSO, DIO, DPO, the cash conversion cycle and inventory build-up."""

from typing import Dict, List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.findings import ForensicFinding, add_finding
from app.agents.specialized.financial_auditor.modules.financial_history import (
    YearFinancialsResult,
    cost_of_revenue_of,
    growth_pct,
)
from app.agents.specialized.financial_auditor.modules.working_capital.schema import YearWorkingCapital
from app.agents.specialized.financial_auditor.modules.working_capital.thresholds import (
    INVENTORY_GROWTH_GAP_THRESHOLD_PP,
    INVENTORY_MIN_GROWTH_PCT,
    WORKING_CAPITAL_DAYS_THRESHOLD_PCT,
)


def measure_working_capital(annual_financials: List[AnnualFinancialInput]) -> List[YearWorkingCapital]:
    """Days of revenue in receivables, days of cost in inventory and payables, per year."""
    rows: List[YearWorkingCapital] = []
    prev_wc: Optional[YearWorkingCapital] = None
    for f in sorted(annual_financials, key=lambda x: x.fiscal_year):
        rev = f.revenue
        cogs = cost_of_revenue_of(f)
        dso = round(f.accounts_receivable / rev * 365.0, 1) if f.accounts_receivable is not None and rev > 0 else None
        dio = round(f.inventories / cogs * 365.0, 1) if f.inventories is not None and cogs and cogs > 0 else None
        dpo = round(f.accounts_payable / cogs * 365.0, 1) if f.accounts_payable is not None and cogs and cogs > 0 else None
        ccc = round((dso or 0.0) + (dio or 0.0) - (dpo or 0.0), 1) if None not in (dso, dpo) else None
        year_wc = YearWorkingCapital(
            fiscal_year=f.fiscal_year,
            dso_days=dso,
            dio_days=dio,
            dpo_days=dpo,
            cash_conversion_cycle_days=ccc,
            dso_change_pct=growth_pct(dso, prev_wc.dso_days) if prev_wc else None,
            dio_change_pct=growth_pct(dio, prev_wc.dio_days) if prev_wc else None,
        )
        rows.append(year_wc)
        prev_wc = year_wc
    return rows


def screen_working_capital(
    annual_inputs: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
    working_capital: List[YearWorkingCapital],
) -> List[ForensicFinding]:
    """INVENTORY_ACCUMULATION (inventory outgrowing revenue) and WORKING_CAPITAL_DETERIORATION (DSO / DIO lengthening)."""
    findings: List[ForensicFinding] = []
    inputs: Dict[int, AnnualFinancialInput] = {f.fiscal_year: f for f in annual_inputs}
    wc = {r.fiscal_year: r for r in working_capital}
    for prev, cur in zip(history, history[1:]):
        p_in, c_in = inputs[prev.fiscal_year], inputs[cur.fiscal_year]
        rev_g = cur.revenue_growth_pct or 0.0
        if p_in.inventories and c_in.inventories:
            inv_g = growth_pct(c_in.inventories, p_in.inventories)
            if inv_g is not None and inv_g - rev_g > INVENTORY_GROWTH_GAP_THRESHOLD_PP and inv_g > INVENTORY_MIN_GROWTH_PCT:
                add_finding(findings, "INVENTORY_ACCUMULATION", "medium", cur.fiscal_year, "inventory_growth_minus_revenue_growth_pp",
                    inv_g - rev_g, INVENTORY_GROWTH_GAP_THRESHOLD_PP,
                    f"INVENTORY ACCUMULATION WARNING (FY{cur.fiscal_year}): Inventory increased {inv_g:+.2f}% "
                    f"(${p_in.inventories:,.1f}M -> ${c_in.inventories:,.1f}M), substantially exceeding revenue "
                    f"growth of {rev_g:+.2f}%; this may indicate obsolescence risk or slowing sell-through.")

    for yr, w in wc.items():
        for metric, change in (("dso", w.dso_change_pct), ("dio", w.dio_change_pct)):
            if change is not None and change > WORKING_CAPITAL_DAYS_THRESHOLD_PCT:
                days = w.dso_days if metric == "dso" else w.dio_days
                add_finding(findings, "WORKING_CAPITAL_DETERIORATION", "low", yr, f"{metric}_change_pct", change,
                    WORKING_CAPITAL_DAYS_THRESHOLD_PCT,
                    f"WORKING CAPITAL DETERIORATION (FY{yr}): {metric.upper()} lengthened {change:.1f}% to "
                    f"{days:.1f} days.")
    return findings
