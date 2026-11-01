"""What the integrity review records: identity checks, completeness, usability, data quality and input corrections."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ExtractionMode = Literal["deterministic", "llm_fallback"]


@dataclass(frozen=True)
class StatementIntegrity:
    """Whether the parsed statements can carry a deterministic audit."""

    identity_checks: List[Dict[str, Any]]
    completeness_score: float
    is_usable: bool

    @property
    def failed_identity_checks(self) -> List[Dict[str, Any]]:
        return [c for c in self.identity_checks if c.get("status") == "failed"]


class AuditDataQuality(BaseModel):
    """Provenance and integrity diagnostics for the numbers in the audit."""

    extraction_mode: ExtractionMode = Field(
        ..., description="deterministic = Python-parsed statements; llm_fallback = LLM-transcribed inputs"
    )
    completeness_score: Optional[float] = Field(
        None, description="Share of institutionally expected latest-year line items that were found (0-1)"
    )
    missing_fields: List[str] = Field(default_factory=list)
    identity_checks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Accounting identity checks (balance sheet, gross profit, NI tie-out)"
    )
    cross_filing_differences: List[Dict[str, Any]] = Field(
        default_factory=list, description="Same field/year reported differently by consecutive filings"
    )
    reconciliation_log: List[str] = Field(
        default_factory=list, description="Every correction, gap-fill or override applied to extracted inputs"
    )
    field_provenance: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="FY<year>.<field> -> source statement, row labels, chunk_id"
    )
    warnings: List[str] = Field(default_factory=list)
    summary_may_be_stale: bool = Field(
        False, description="True when inputs were corrected after the LLM wrote the auditor summary"
    )


class InputCorrection(BaseModel):
    """A cited override of one deterministically extracted line item."""

    target: Literal["annual", "balance_sheet", "prior_balance_sheet"]
    fiscal_year: int
    field: str
    value: float
    reason: str
    source_chunk_id: Optional[str] = None
