"""Undervalued, fairly valued or overvalued? The base-case upside against the share price, read against a
+/-10% band (a valuation stance, not a Buy/Sell rating)."""

from typing import Optional

from app.agents.specialized.valuation_specialist.modules.valuation_stance.schema import ValuationStance, ValuationStanceResult
from app.agents.specialized.valuation_specialist.modules.valuation_stance.thresholds import STANCE_BAND_PCT
from app.agents.specialized.valuation_specialist.upside import upside_pct


def _stance_for(upside: Optional[float]) -> Optional[ValuationStance]:
    if upside is None:
        return None
    if upside > STANCE_BAND_PCT:
        return "Undervalued"
    if upside < -STANCE_BAND_PCT:
        return "Overvalued"
    return "Fairly Valued"


def take_valuation_stance(fair_value_per_share: float, price: Optional[float]) -> ValuationStanceResult:
    """Base-case upside against the share price; beyond the band it is a call, inside it is fair value."""
    upside = upside_pct(fair_value_per_share, price)
    return ValuationStanceResult(current_share_price=price, upside_downside_pct=upside, valuation_stance=_stance_for(upside))
