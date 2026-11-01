"""One-off adjustments and the normalized earnings they produce."""

from dataclasses import dataclass
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NormalizationAdjustment(BaseModel):
    """
    A non-recurring or non-core item to strip from reported earnings.

    - `amount`: positive magnitude in $ millions, exactly as disclosed.
    - `direction`: "inflated_reported_earnings" for gains, credit sales, tax benefits and
      valuation-allowance releases (normalized earnings are LOWER); "depressed_reported_earnings"
      for charges such as impairments, restructuring and litigation (normalized earnings are HIGHER).
    """

    fiscal_year: int
    label: str
    amount: float = Field(..., gt=0, description="Positive magnitude in $ millions")
    direction: Literal["inflated_reported_earnings", "depressed_reported_earnings"]
    affects: Literal["operating_income", "net_income"] = "operating_income"
    rationale: Optional[str] = None
    source_chunk_id: Optional[str] = None

    @property
    def signed_effect(self) -> float:
        """Effect on reported earnings (+ inflated, - depressed); normalized = reported - signed_effect."""
        return self.amount if self.direction == "inflated_reported_earnings" else -self.amount


@dataclass(frozen=True)
class NormalizedYear:
    fiscal_year: int
    non_operating_income: Optional[float]
    non_operating_to_operating_income_pct: Optional[float]
    adjustments: List[NormalizationAdjustment]
    operating_income_adjustments_total: float
    net_income_adjustments_total: float
    adjustments_pct_of_operating_income: Optional[float]
    normalized_operating_income: float
    normalized_operating_margin_pct: float
    normalized_net_income: float
    normalized_net_margin_pct: float
