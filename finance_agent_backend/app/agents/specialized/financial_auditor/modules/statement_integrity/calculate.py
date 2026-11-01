"""Can I trust the extracted numbers? Accounting identities, completeness, usability, restatements, data quality."""

from typing import Any, Dict, List, Optional

from app.agents.specialized.financial_auditor.context.extract import (
    REQUIRED_BALANCE_FIELDS,
    SHARE_COUNT_FIELDS,
    ExtractionResult,
    percent_difference,
)
from app.agents.specialized.financial_auditor.findings import ForensicFinding
from app.agents.specialized.financial_auditor.modules.statement_integrity.schema import (
    AuditDataQuality,
    StatementIntegrity,
)
from app.agents.specialized.financial_auditor.modules.statement_integrity.thresholds import (
    EXPECTED_LATEST_FIELDS,
    IDENTITY_TOLERANCE_PCT,
)


def _identity(name: str, lhs: float, rhs: float, critical: bool, detail: str) -> Dict[str, Any]:
    diff = percent_difference(lhs, rhs)
    return {
        "name": name,
        "status": "passed" if diff <= IDENTITY_TOLERANCE_PCT else "failed",
        "critical": critical,
        "lhs": round(lhs, 2),
        "rhs": round(rhs, 2),
        "difference_pct": round(diff, 3),
        "detail": detail,
    }


def check_accounting_identities(
    annual: Dict[int, Dict[str, float]],
    balance: Dict[int, Dict[str, float]],
    years: List[int],
    fiscal_year: int,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    bs = balance.get(fiscal_year, {})
    ta = bs.get("total_assets")
    if ta is not None and bs.get("total_liabilities_and_equity") is not None:
        checks.append(_identity(
            f"FY{fiscal_year} balance sheet: total assets = total liabilities and equity",
            ta, bs["total_liabilities_and_equity"], True,
            "Validates that the balance sheet grid was parsed with correct period alignment.",
        ))
    if ta is not None and bs.get("total_liabilities") is not None and bs.get("stockholders_equity") is not None:
        rhs = bs["total_liabilities"] + bs["stockholders_equity"] + (bs.get("noncontrolling_interests") or 0.0)
        checks.append(_identity(
            f"FY{fiscal_year} balance sheet: assets = liabilities + mezzanine + equity (incl. NCI)",
            ta, rhs, False,
            "Validates the stockholders' equity and liabilities rows used for ROE / invested capital.",
        ))
    for yr in years:
        vals = annual.get(yr, {})
        if None not in (vals.get("revenue"), vals.get("cost_of_revenue"), vals.get("gross_profit")):
            checks.append(_identity(
                f"FY{yr} gross profit = revenue - cost of revenue",
                vals["gross_profit"], vals["revenue"] - vals["cost_of_revenue"], False,
                "Validates revenue and cost of revenue row selection.",
            ))
        if vals.get("net_income_total") is not None and vals.get("net_income_cf") is not None:
            checks.append(_identity(
                f"FY{yr} net income: income statement = cash flow statement",
                vals["net_income_total"], vals["net_income_cf"], False,
                "Validates that the income statement and cash flow statement describe the same periods.",
            ))
    return checks


def score_completeness(extraction: ExtractionResult) -> float:
    """Share of the institutionally expected latest-year line items that were found."""
    latest = extraction.annual.get(extraction.fiscal_year, {})
    bs = extraction.balance.get(extraction.fiscal_year, {})
    found = 0
    for f in EXPECTED_LATEST_FIELDS:
        if f == "gross_profit_or_cost_of_revenue":
            present = latest.get("gross_profit") is not None or latest.get("cost_of_revenue") is not None
        else:
            present = latest.get(f) is not None or bs.get(f) is not None
        found += 1 if present else 0
    return round(found / len(EXPECTED_LATEST_FIELDS), 3)


def _is_usable(extraction: ExtractionResult, identity_checks: List[Dict[str, Any]]) -> bool:
    """True when every required field exists for >= 2 contiguous years ending at the target year."""
    if len(extraction.years) < 2 or extraction.years[-1] != extraction.fiscal_year:
        return False
    latest_bs = extraction.balance.get(extraction.fiscal_year, {})
    if any(latest_bs.get(f) is None for f in REQUIRED_BALANCE_FIELDS):
        return False
    if all(latest_bs.get(f) is None for f in SHARE_COUNT_FIELDS):
        return False
    failed = [c for c in identity_checks if c.get("status") == "failed"]
    return not any(c.get("critical") for c in failed)


def assess_statement_integrity(extraction: ExtractionResult) -> StatementIntegrity:
    """Identity checks, completeness and whether the parsed statements can carry a deterministic audit."""
    checks = check_accounting_identities(extraction.annual, extraction.balance, extraction.years, extraction.fiscal_year)
    return StatementIntegrity(
        identity_checks=checks,
        completeness_score=score_completeness(extraction),
        is_usable=_is_usable(extraction, checks),
    )


def summarize_for_judge(extraction: ExtractionResult, integrity: StatementIntegrity) -> Dict[str, Any]:
    """The extraction diagnostics the auditor LLM reviews."""
    return {
        "years": extraction.years,
        "completeness_score": integrity.completeness_score,
        "missing_fields": extraction.missing_fields,
        "failed_identity_checks": integrity.failed_identity_checks,
        "cross_filing_differences": extraction.cross_filing_differences,
        "warnings": extraction.warnings,
    }


def note_restatements(
    extraction: ExtractionResult,
    prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]],
) -> List[str]:
    """Cross-filing restatement notes, or why there are none."""
    notes = [
        f"FY{d['fiscal_year']} {d['field'].replace('_', ' ')} ({d['statement']}): "
        f"{d['older_value']:,.1f} in the FY{d['older_filing_year']} 10-K vs {d['latest_value']:,.1f} in the "
        f"FY{d['latest_filing_year']} 10-K ({d['difference_pct']:.2f}% difference); latest filing figure used."
        for d in extraction.cross_filing_differences
    ]
    if notes:
        return notes
    filings = sorted(
        {p.get("fiscal_year") for items in (prefetched_tables or {}).values() for p in items if p.get("fiscal_year")},
        reverse=True,
    )
    if len(filings) > 1:
        return [
            "No retrospective restatements identified: overlapping years agree across the "
            + ", ".join(f"FY{y}" for y in filings) + " 10-K filings for all extracted line items."
        ]
    return ["Single filing available; cross-filing restatement review not possible."]


def record_data_quality(
    extraction: Optional[ExtractionResult],
    integrity: Optional[StatementIntegrity],
    mode: str,
    log: List[str],
    stale: bool,
) -> AuditDataQuality:
    if extraction is None or integrity is None:
        return AuditDataQuality(
            extraction_mode=mode,
            reconciliation_log=log,
            warnings=["Primary statements could not be retrieved or parsed; inputs transcribed by the LLM."],
            summary_may_be_stale=stale,
        )
    return AuditDataQuality(
        extraction_mode=mode,
        completeness_score=integrity.completeness_score,
        missing_fields=extraction.missing_fields,
        identity_checks=integrity.identity_checks,
        cross_filing_differences=extraction.cross_filing_differences,
        reconciliation_log=log,
        field_provenance=extraction.field_provenance,
        warnings=extraction.warnings,
        summary_may_be_stale=stale,
    )


def flag_data_quality(dq: AuditDataQuality) -> List[ForensicFinding]:
    """DATA_QUALITY findings: each failed identity check, and LLM-transcribed inputs."""
    findings: List[ForensicFinding] = []
    for check in dq.identity_checks:
        if check.get("status") == "failed":
            findings.append(ForensicFinding(
                code="DATA_QUALITY", severity="high" if check.get("critical") else "medium",
                metric="identity_check_difference_pct", value=check.get("difference_pct"),
                threshold=None, message=f"DATA QUALITY: Identity check failed — {check['name']} "
                f"({check['lhs']:,.1f} vs {check['rhs']:,.1f}, {check['difference_pct']:.2f}% apart).",
            ))
    if dq.extraction_mode == "llm_fallback":
        findings.append(ForensicFinding(
            code="DATA_QUALITY", severity="info", metric="extraction_mode", threshold=None,
            message="DATA QUALITY: Statements were not machine-parsed; inputs were transcribed by the LLM "
                    "and gap-filled where parsed values existed. Verify key figures against the filing.",
        ))
    return findings
