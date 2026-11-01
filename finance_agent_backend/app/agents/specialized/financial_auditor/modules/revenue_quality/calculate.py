"""Is reported revenue real and collectable? Receivables growth against revenue growth."""

from typing import Dict, List

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.findings import ForensicFinding, add_finding
from app.agents.specialized.financial_auditor.modules.financial_history import YearFinancialsResult, growth_pct
from app.agents.specialized.financial_auditor.modules.revenue_quality.thresholds import (
    RECEIVABLES_GROWTH_GAP_THRESHOLD_PP,
    RECEIVABLES_MIN_GROWTH_PCT,
)


def screen_revenue_quality(
    annual_inputs: List[AnnualFinancialInput],
    history: List[YearFinancialsResult],
) -> List[ForensicFinding]:
    """RECEIVABLES_DIVERGENCE: receivables growing well ahead of revenue."""
    findings: List[ForensicFinding] = []
    inputs: Dict[int, AnnualFinancialInput] = {f.fiscal_year: f for f in annual_inputs}
    for prev, cur in zip(history, history[1:]):
        p_in, c_in = inputs[prev.fiscal_year], inputs[cur.fiscal_year]
        rev_g = cur.revenue_growth_pct or 0.0
        if p_in.accounts_receivable and c_in.accounts_receivable is not None:
            ar_g = growth_pct(c_in.accounts_receivable, p_in.accounts_receivable)
            if ar_g is not None and ar_g - rev_g > RECEIVABLES_GROWTH_GAP_THRESHOLD_PP and ar_g > RECEIVABLES_MIN_GROWTH_PCT:
                add_finding(findings, "RECEIVABLES_DIVERGENCE", "medium", cur.fiscal_year, "ar_growth_minus_revenue_growth_pp",
                    ar_g - rev_g, RECEIVABLES_GROWTH_GAP_THRESHOLD_PP,
                    f"RECEIVABLES DIVERGENCE (FY{cur.fiscal_year}): Accounts receivable grew {ar_g:+.2f}% "
                    f"(${p_in.accounts_receivable:,.1f}M -> ${c_in.accounts_receivable:,.1f}M) versus revenue "
                    f"growth of {rev_g:+.2f}%, which may signal aggressive revenue recognition or collection delays.")
    return findings
