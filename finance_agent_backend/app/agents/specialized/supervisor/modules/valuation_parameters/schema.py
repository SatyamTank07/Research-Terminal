"""Valuation inputs the user set in chat, and the record the session remembers for one ticker."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

SESSION_KEY = "valuation_overrides"
OVERRIDE_FIELDS = ("wacc", "terminal_growth", "beta", "share_price")


class ValuationOverrides(BaseModel):
    """User-specified valuation parameters (decimals) with the notes explaining anything rejected."""

    wacc: Optional[float] = None
    terminal_growth: Optional[float] = None
    beta: Optional[float] = None
    share_price: Optional[float] = None
    reset: bool = Field(False, description="The user asked to drop previously remembered overrides")
    notes: List[str] = Field(default_factory=list)

    def values(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in OVERRIDE_FIELDS if getattr(self, k) is not None}


@dataclass(frozen=True)
class ValuationParameters:
    """The record to apply and remember ({"ticker", ...values} or empty) and the carry-over note, if any."""

    overrides: Dict[str, Any]
    carry_note: Optional[str]
