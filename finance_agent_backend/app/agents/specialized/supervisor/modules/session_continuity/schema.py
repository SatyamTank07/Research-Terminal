"""The filing a research turn ran on, as the session remembers it."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SessionFiling:
    ticker: str
    company_name: str
    fiscal_year: int
    available_fiscal_years: List[int]
