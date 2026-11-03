"""The fiscal year a research turn asks for, and the 10-K year the catalog can serve."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RequestedYear:
    """The year to look up (None = latest) and the note when it was carried from the session."""

    year: Optional[int]
    note: Optional[str]


@dataclass(frozen=True)
class FilingChoice:
    """The year the user asked for, the 10-K year used, and whether the latest filing stood in for it."""

    year_requested: Optional[int]
    fiscal_year: int
    year_substituted: bool
