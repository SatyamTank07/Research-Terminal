"""Screens the auditor's proposed one-off adjustments: only discrete, non-recurring items are accepted."""

import re
from typing import Any, Dict, List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.earnings_normalization.schema import NormalizationAdjustment

_AGGREGATE_LINE_LABEL = re.compile(
    r"non[- ]?operating|non[- ]?core income|other income|other \(expense\)|interest (and other )?income|"
    r"total other|income dependence"
)
_RECURRING_COST_LABEL = re.compile(
    r"research and development|selling, general|general and administrative|stock[- ]based|share[- ]based"
)
_TAX_LABEL = re.compile(r"\btax(?:es)?\b")
_DISCRETE_TAX_EVENT = re.compile(
    r"valuation[- ]allowance|remeasure|re-measure|settlement|tax (?:act|reform|law)|tax cuts|tcja|enact|"
    r"repatriat|transition tax|one[- ]time|discrete|intra[- ]entity|ip transfer|uncertain tax|statute"
)


def screen_normalization_adjustments(
    annual_financials: List[AnnualFinancialInput],
    adjustments: Optional[List[NormalizationAdjustment]],
) -> Dict[str, List[Any]]:
    """
    Deterministically rejects normalization items that are not discrete non-recurring disclosures:
    out-of-window years, aggregate non-operating subtotals (already measured by the
    NON_CORE_INCOME_DEPENDENCE screen), recurring operating costs and the recurring income tax
    provision (only discrete tax events such as valuation-allowance releases qualify).
    """
    inputs = {f.fiscal_year: f for f in annual_financials}
    accepted: List[NormalizationAdjustment] = []
    rejected: List[Dict[str, Any]] = []
    for adj in adjustments or []:
        reason = None
        f = inputs.get(adj.fiscal_year)
        label = adj.label.lower()
        tax_provision = abs(f.income_tax_expense) if f is not None and f.income_tax_expense else None
        if f is None:
            reason = f"FY{adj.fiscal_year} is outside the audited window {sorted(inputs)}"
        elif _AGGREGATE_LINE_LABEL.search(label):
            reason = "aggregate non-operating line; measured deterministically by NON_CORE_INCOME_DEPENDENCE"
        elif _RECURRING_COST_LABEL.search(label):
            reason = "recurring operating cost; not a non-recurring item"
        elif _TAX_LABEL.search(label) and not _DISCRETE_TAX_EVENT.search(f"{label} {(adj.rationale or '').lower()}"):
            reason = (
                "recurring income tax provision; only discrete tax events (valuation-allowance release, "
                "remeasurement, settlement, tax-law change) qualify, and the effective tax rate is measured separately"
            )
        elif tax_provision and abs(adj.amount - tax_provision) <= 0.005 * tax_provision:
            reason = "amount equals the full income tax provision, a recurring expense"
        elif f.pretax_income is not None:
            non_op = f.pretax_income - f.operating_income
            if non_op and abs(adj.amount - abs(non_op)) <= 0.005 * abs(non_op):
                reason = "amount equals the aggregate non-operating income subtotal"
        if reason:
            rejected.append({**adj.model_dump(), "rejection": reason})
        else:
            accepted.append(adj)
    return {"accepted": accepted, "rejected": rejected}
