"""The tax rate each year was taxed at, and the latest rate the returns measures use."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaxYear:
    fiscal_year: int
    effective_tax_rate_pct: Optional[float]
    deferred_tax_to_net_income_pct: Optional[float]


@dataclass(frozen=True)
class LatestTaxRate:
    """The latest year's effective tax rate (decimal), or the statutory rate and why."""

    rate: float
    source: str
