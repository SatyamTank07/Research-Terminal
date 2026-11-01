"""The forensic finding every audit module emits, and the rule that ranks them for the report.

Findings rank by severity, then newest fiscal year first. Findings that tie on both keep the order of
`FINDING_ORDER`, the order in which the audit's screens have always run.
"""

from typing import Any, Iterable, List, Literal, Optional

from pydantic import BaseModel

Severity = Literal["high", "medium", "low", "info"]
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}

FINDING_ORDER = (
    "EARNINGS_QUALITY_DIVERGENCE",
    "WEAK_CASH_CONVERSION",
    "CHRONIC_ACCRUAL_DEFICIT",
    "NEGATIVE_BOOK_EQUITY",
    "RECEIVABLES_DIVERGENCE",
    "INVENTORY_ACCUMULATION",
    "NON_CORE_INCOME_DEPENDENCE",
    "TAX_ANOMALY",
    "SBC_INTENSITY",
    "HIGH_ACCRUALS",
    "CAPEX_DRIVEN_FCF",
    "UNDERINVESTMENT",
    "SHARE_DILUTION",
    "NEGATIVE_OPERATING_LEVERAGE",
    "MARGIN_COMPRESSION",
    "WORKING_CAPITAL_DETERIORATION",
    "DATA_QUALITY",
)


class ForensicFinding(BaseModel):
    code: str
    severity: Severity
    fiscal_year: Optional[int] = None
    metric: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    message: str


def add_finding(
    findings: List[ForensicFinding],
    code: str,
    severity: Severity,
    year: Optional[int],
    metric: str,
    value: Optional[float],
    threshold: Optional[float],
    message: str,
) -> None:
    """Appends one finding with its value rounded to 2 dp."""
    findings.append(ForensicFinding(
        code=code, severity=severity, fiscal_year=year, metric=metric,
        value=round(value, 2) if value is not None else None, threshold=threshold, message=message,
    ))


def rank_findings(findings: Iterable[ForensicFinding]) -> List[ForensicFinding]:
    """Severity first, newest year next, then the screen order; equal findings keep their order."""
    def rank(f: ForensicFinding) -> Any:
        order = FINDING_ORDER.index(f.code) if f.code in FINDING_ORDER else len(FINDING_ORDER)
        return SEVERITY_RANK[f.severity], -(f.fiscal_year or 0), order

    return sorted(findings, key=rank)
