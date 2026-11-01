"""Is the tax rate sustainable? Effective rates, deferred tax against net income, and the rate to tax-effect with."""

from typing import List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.findings import ForensicFinding, Severity, add_finding
from app.agents.specialized.financial_auditor.modules.financial_history import pct_of
from app.agents.specialized.financial_auditor.modules.tax_quality.schema import LatestTaxRate, TaxYear
from app.agents.specialized.financial_auditor.modules.tax_quality.thresholds import (
    DEFERRED_TAX_TO_NI_THRESHOLD_PCT,
    ETR_HIGH_PCT,
    ETR_LOW_PCT,
    STATUTORY_TAX_RATE,
)


def derived_tax_rate(inp: AnnualFinancialInput) -> Optional[float]:
    """Derived effective tax rate (decimal) when pre-tax income is positive and tax is an expense."""
    if inp.pretax_income is None or inp.income_tax_expense is None:
        return None
    if inp.pretax_income <= 0 or inp.income_tax_expense < 0:
        return None
    return inp.income_tax_expense / inp.pretax_income


def tax_effect_rate(inp: AnnualFinancialInput) -> float:
    """The rate a pre-tax adjustment is tax-effected at: the year's derived rate, else statutory."""
    year_rate = derived_tax_rate(inp)
    return year_rate if year_rate is not None else STATUTORY_TAX_RATE


def latest_tax_rate(latest_input: AnnualFinancialInput) -> LatestTaxRate:
    """Effective tax rate with provenance: derived from the 10-K, else statutory with the reason."""
    tax_rate, tax_source = STATUTORY_TAX_RATE, "statutory_default_not_provided"
    if latest_input.pretax_income is not None and latest_input.income_tax_expense is not None:
        if latest_input.pretax_income <= 0:
            tax_source = "statutory_fallback_due_to_pretax_loss"
        elif latest_input.income_tax_expense < 0:
            tax_source = "statutory_fallback_due_to_tax_benefit"
        else:
            tax_rate = round(latest_input.income_tax_expense / latest_input.pretax_income, 4)
            tax_source = "derived_from_10k"
    return LatestTaxRate(rate=tax_rate, source=tax_source)


def measure_tax_quality(annual_financials: List[AnnualFinancialInput]) -> List[TaxYear]:
    """Effective tax rate and deferred tax as a share of net income, per year."""
    rows: List[TaxYear] = []
    for f in sorted(annual_financials, key=lambda x: x.fiscal_year):
        etr_pct = pct_of(f.income_tax_expense, f.pretax_income) if (
            f.pretax_income not in (None, 0) and f.income_tax_expense is not None
        ) else None
        rows.append(TaxYear(
            fiscal_year=f.fiscal_year,
            effective_tax_rate_pct=etr_pct,
            deferred_tax_to_net_income_pct=pct_of(f.deferred_income_taxes, abs(f.net_income)) if f.net_income else None,
        ))
    return rows


def screen_tax_quality(annual_inputs: List[AnnualFinancialInput], tax_years: List[TaxYear]) -> List[ForensicFinding]:
    """TAX_ANOMALY: a tax benefit on pre-tax profit, an effective rate outside 5-40%, or large deferred tax."""
    findings: List[ForensicFinding] = []
    inputs = {f.fiscal_year: f for f in annual_inputs}
    for q in tax_years:
        yr = q.fiscal_year
        f_in = inputs[yr]
        tax_reasons = []
        severity: Severity = "medium"
        if f_in.income_tax_expense is not None and f_in.pretax_income is not None and f_in.pretax_income > 0:
            if f_in.income_tax_expense < 0:
                severity = "high"
                tax_reasons.append(
                    f"a ${abs(f_in.income_tax_expense):,.1f}M tax benefit on positive pre-tax income of "
                    f"${f_in.pretax_income:,.1f}M")
            elif q.effective_tax_rate_pct is not None and not (ETR_LOW_PCT <= q.effective_tax_rate_pct <= ETR_HIGH_PCT):
                tax_reasons.append(f"an effective tax rate of {q.effective_tax_rate_pct:.1f}%")
        if q.deferred_tax_to_net_income_pct is not None and \
                abs(q.deferred_tax_to_net_income_pct) > DEFERRED_TAX_TO_NI_THRESHOLD_PCT:
            tax_reasons.append(
                f"a deferred tax adjustment of ${f_in.deferred_income_taxes:,.1f}M "
                f"({q.deferred_tax_to_net_income_pct:.1f}% of net income)")
        if tax_reasons:
            add_finding(findings, "TAX_ANOMALY", severity, yr, "effective_tax_rate_pct", q.effective_tax_rate_pct, None,
                f"TAX ANOMALY (FY{yr}): Reported earnings include {'; '.join(tax_reasons)}. Net income and margins "
                f"for this year are not comparable without normalization (e.g. valuation-allowance release).")
    return findings
