"""What are earnings without one-offs? Non-operating income, accepted adjustments and normalized EBIT / net income."""

from typing import Dict, List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.findings import ForensicFinding, add_finding
from app.agents.specialized.financial_auditor.modules.earnings_normalization.schema import NormalizationAdjustment, NormalizedYear
from app.agents.specialized.financial_auditor.modules.earnings_normalization.thresholds import (
    NON_OPERATING_TO_EBIT_THRESHOLD_PCT,
    NORMALIZATION_TO_EBIT_THRESHOLD_PCT,
)
from app.agents.specialized.financial_auditor.modules.financial_history import YearFinancialsResult, pct_of
from app.agents.specialized.financial_auditor.modules.tax_quality import tax_effect_rate


def normalize_earnings(
    annual_financials: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
    adjustments: Optional[List[NormalizationAdjustment]] = None,
) -> List[NormalizedYear]:
    """Strips accepted one-offs from reported EBIT and net income; pre-tax items are tax-effected at the year's rate."""
    hist_map = {h.fiscal_year: h for h in history}
    adj_by_year: Dict[int, List[NormalizationAdjustment]] = {}
    for adj in adjustments or []:
        if adj.fiscal_year not in hist_map:
            raise ValueError(
                f"normalization adjustment '{adj.label}' targets FY{adj.fiscal_year}, "
                f"outside the audited window {sorted(hist_map)}."
            )
        adj_by_year.setdefault(adj.fiscal_year, []).append(adj)

    rows: List[NormalizedYear] = []
    for f in sorted(annual_financials, key=lambda x: x.fiscal_year):
        rev = f.revenue
        non_op = round(f.pretax_income - f.operating_income, 2) if f.pretax_income is not None else None
        year_adj = adj_by_year.get(f.fiscal_year, [])
        ebit_adj = round(sum(a.signed_effect for a in year_adj if a.affects == "operating_income"), 2)
        ni_direct_adj = round(sum(a.signed_effect for a in year_adj if a.affects == "net_income"), 2)
        after_tax_factor = 1.0 - tax_effect_rate(f)
        ni_total_adj = round(ebit_adj * after_tax_factor + ni_direct_adj, 2)
        norm_ebit = round(f.operating_income - ebit_adj, 2)
        norm_ni = round(f.net_income - ni_total_adj, 2)
        abs_ebit_adj = sum(abs(a.amount) for a in year_adj if a.affects == "operating_income")
        rows.append(NormalizedYear(
            fiscal_year=f.fiscal_year,
            non_operating_income=non_op,
            non_operating_to_operating_income_pct=pct_of(non_op, f.operating_income) if f.operating_income > 0 else None,
            adjustments=year_adj,
            operating_income_adjustments_total=ebit_adj,
            net_income_adjustments_total=ni_total_adj,
            adjustments_pct_of_operating_income=pct_of(abs_ebit_adj, abs(f.operating_income)) if year_adj else None,
            normalized_operating_income=norm_ebit,
            normalized_operating_margin_pct=pct_of(norm_ebit, rev) or 0.0,
            normalized_net_income=norm_ni,
            normalized_net_margin_pct=pct_of(norm_ni, rev) or 0.0,
        ))
    return rows


def screen_earnings_normalization(normalized: List[NormalizedYear]) -> List[ForensicFinding]:
    """NON_CORE_INCOME_DEPENDENCE: non-operating income, or accepted one-offs, above their share-of-EBIT thresholds."""
    findings: List[ForensicFinding] = []
    for q in normalized:
        yr = q.fiscal_year
        if q.non_operating_to_operating_income_pct is not None and \
                q.non_operating_to_operating_income_pct > NON_OPERATING_TO_EBIT_THRESHOLD_PCT:
            add_finding(findings, "NON_CORE_INCOME_DEPENDENCE", "medium", yr, "non_operating_to_operating_income_pct",
                q.non_operating_to_operating_income_pct, NON_OPERATING_TO_EBIT_THRESHOLD_PCT,
                f"NON-CORE INCOME DEPENDENCE (FY{yr}): Non-operating income of ${q.non_operating_income:,.1f}M "
                f"equals {q.non_operating_to_operating_income_pct:.1f}% of operating income; pre-tax earnings "
                f"lean on interest and other income rather than the core business.")
        if q.adjustments_pct_of_operating_income is not None and \
                q.adjustments_pct_of_operating_income > NORMALIZATION_TO_EBIT_THRESHOLD_PCT:
            labels = ", ".join(a.label for a in q.adjustments if a.affects == "operating_income")
            add_finding(findings, "NON_CORE_INCOME_DEPENDENCE", "high", yr, "normalization_adjustments_pct_of_operating_income",
                q.adjustments_pct_of_operating_income, NORMALIZATION_TO_EBIT_THRESHOLD_PCT,
                f"NORMALIZATION MATERIALITY (FY{yr}): Non-recurring / non-core items ({labels}) equal "
                f"{q.adjustments_pct_of_operating_income:.1f}% of reported operating income; normalized operating "
                f"income is ${q.normalized_operating_income:,.1f}M ({q.normalized_operating_margin_pct:.2f}% margin).")
    return findings
