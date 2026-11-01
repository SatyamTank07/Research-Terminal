"""Are profits backed by cash? Accruals, the Sloan ratio, FCF conversion and stock-based compensation."""

from typing import List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.findings import ForensicFinding, add_finding
from app.agents.specialized.financial_auditor.modules.cash_conversion.schema import CashConversionYear
from app.agents.specialized.financial_auditor.modules.cash_conversion.thresholds import (
    FCF_CONVERSION_WEAK_PCT,
    SBC_TO_EBIT_THRESHOLD_PCT,
    SBC_TO_OCF_THRESHOLD_PCT,
    SLOAN_ACCRUALS_THRESHOLD_PCT,
)
from app.agents.specialized.financial_auditor.modules.financial_history import (
    YearFinancialsResult,
    growth_pct,
    pct_of,
)


def measure_cash_conversion(
    annual_financials: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
    prior_total_assets: Optional[float] = None,
) -> List[CashConversionYear]:
    """Accruals (net income minus OCF) over average total assets, and SBC against OCF, EBIT and FCF, per year."""
    hist_map = {h.fiscal_year: h for h in history}
    rows: List[CashConversionYear] = []
    prev: Optional[AnnualFinancialInput] = None
    for f in sorted(annual_financials, key=lambda x: x.fiscal_year):
        h = hist_map[f.fiscal_year]
        ni_for_accruals = f.net_income_total if f.net_income_total is not None else f.net_income
        accruals = round(ni_for_accruals - f.operating_cash_flow, 2)
        ta_prev = prev.total_assets if prev is not None else prior_total_assets
        avg_ta = None
        if f.total_assets is not None:
            avg_ta = (f.total_assets + ta_prev) / 2.0 if ta_prev is not None else f.total_assets
        sbc = f.stock_based_compensation
        rows.append(CashConversionYear(
            fiscal_year=f.fiscal_year,
            accruals=accruals,
            sloan_accruals_ratio_pct=pct_of(accruals, avg_ta),
            stock_based_compensation=sbc,
            sbc_to_ocf_pct=pct_of(sbc, f.operating_cash_flow) if f.operating_cash_flow > 0 else None,
            sbc_to_operating_income_pct=pct_of(sbc, f.operating_income) if f.operating_income > 0 else None,
            sbc_adjusted_fcf=round(h.free_cash_flow - sbc, 2) if sbc is not None else None,
        ))
        prev = f
    return rows


def screen_cash_conversion(
    history: List[YearFinancialsResult],
    cash_conversion: List[CashConversionYear],
) -> List[ForensicFinding]:
    """EARNINGS_QUALITY_DIVERGENCE, WEAK_CASH_CONVERSION, CHRONIC_ACCRUAL_DEFICIT, SBC_INTENSITY and HIGH_ACCRUALS."""
    findings: List[ForensicFinding] = []
    # Net income up while OCF down
    for prev, cur in zip(history, history[1:]):
        ni_g = growth_pct(cur.net_income, prev.net_income)
        ocf_g = growth_pct(cur.operating_cash_flow, prev.operating_cash_flow)
        if ni_g is not None and ocf_g is not None and ni_g > 0.0 and ocf_g < 0.0:
            add_finding(findings, "EARNINGS_QUALITY_DIVERGENCE", "high", cur.fiscal_year, "net_income_vs_ocf_growth_pct", ni_g, 0.0,
                f"EARNINGS QUALITY DIVERGENCE (FY{cur.fiscal_year}): Net income increased by +{ni_g:.2f}% "
                f"(${prev.net_income:,.1f}M -> ${cur.net_income:,.1f}M) while operating cash flow declined by "
                f"{ocf_g:.2f}% (${prev.operating_cash_flow:,.1f}M -> ${cur.operating_cash_flow:,.1f}M), indicating "
                f"accrual expansion, non-operating gains, or tax/working-capital cash timing outflows.")

    profitable = [h for h in history if h.net_income > 0 and h.fcf_conversion_pct is not None]
    if len(profitable) >= 2 and all(h.fcf_conversion_pct < FCF_CONVERSION_WEAK_PCT for h in profitable):
        avg_conv = sum(h.fcf_conversion_pct for h in profitable) / len(profitable)
        add_finding(findings, "WEAK_CASH_CONVERSION", "medium", history[-1].fiscal_year, "avg_fcf_conversion_pct", avg_conv,
            FCF_CONVERSION_WEAK_PCT,
            f"WEAK CASH CONVERSION WARNING: Free cash flow conversion (FCF / Net Income) has been chronically "
            f"weak (< {FCF_CONVERSION_WEAK_PCT:.0f}%) across all reported years with positive earnings (averaging {avg_conv:.1f}%), "
            f"signaling high capital intensity or poor earnings-to-cash realization.")

    if len(history) >= 2 and all(h.operating_cash_flow < h.net_income for h in history):
        add_finding(findings, "CHRONIC_ACCRUAL_DEFICIT", "high", history[-1].fiscal_year, "ocf_below_net_income_years",
            float(len(history)), None,
            "CHRONIC ACCRUAL DEFICIT: Operating cash flow was strictly lower than net income in every "
            "reported fiscal year, signaling persistently low cash conversion quality.")

    for yr, q in {r.fiscal_year: r for r in cash_conversion}.items():
        if q.sbc_to_ocf_pct is not None and (
            q.sbc_to_ocf_pct > SBC_TO_OCF_THRESHOLD_PCT
            or (q.sbc_to_operating_income_pct or 0.0) > SBC_TO_EBIT_THRESHOLD_PCT
        ):
            add_finding(findings, "SBC_INTENSITY", "medium", yr, "sbc_to_ocf_pct", q.sbc_to_ocf_pct, SBC_TO_OCF_THRESHOLD_PCT,
                f"STOCK-BASED COMPENSATION INTENSITY (FY{yr}): SBC of ${q.stock_based_compensation:,.1f}M equals "
                f"{q.sbc_to_ocf_pct:.1f}% of operating cash flow and "
                f"{q.sbc_to_operating_income_pct or 0.0:.1f}% of operating income; SBC-adjusted FCF is "
                f"${q.sbc_adjusted_fcf:,.1f}M.")

        if q.sloan_accruals_ratio_pct is not None and q.sloan_accruals_ratio_pct > SLOAN_ACCRUALS_THRESHOLD_PCT:
            add_finding(findings, "HIGH_ACCRUALS", "medium", yr, "sloan_accruals_ratio_pct", q.sloan_accruals_ratio_pct,
                SLOAN_ACCRUALS_THRESHOLD_PCT,
                f"HIGH ACCRUALS (FY{yr}): Sloan accruals ratio of {q.sloan_accruals_ratio_pct:.1f}% of average total "
                f"assets; earnings are running materially ahead of operating cash flow.")
    return findings
