"""Result types of the valuation-stance module."""

from dataclasses import dataclass
from typing import Literal, Optional

ValuationStance = Literal["Undervalued", "Fairly Valued", "Overvalued"]


@dataclass(frozen=True)
class ValuationStanceResult:
    """The base-case fair value against the share price: upside in percent and the stance it implies."""

    current_share_price: Optional[float]
    upside_downside_pct: Optional[float]
    valuation_stance: Optional[ValuationStance]
