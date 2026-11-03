"""Which 10-K: the requested year, else the session's year on a follow-up, else the latest; substitution flagged."""

from app.agents.specialized.supervisor.modules.filing_resolution.calculate import (
    choose_filing_year,
    explain_missing_filing,
    extract_fiscal_year,
    names_entity,
    resolve_requested_year,
)
from app.agents.specialized.supervisor.modules.filing_resolution.schema import FilingChoice, RequestedYear

__all__ = [
    "FilingChoice",
    "RequestedYear",
    "choose_filing_year",
    "explain_missing_filing",
    "extract_fiscal_year",
    "names_entity",
    "resolve_requested_year",
]
