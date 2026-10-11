"""Deterministic extraction of user-specified valuation parameters and their session carry-over.

Parses a discount rate (WACC), terminal growth, beta and share price from free text. Each value is
range-checked; anything implausible is ignored with a note instead of reaching the DCF engine.

Overrides are remembered in `session_state["valuation_overrides"]`, bound to the ticker they were
applied to. A value given before a company is named ("Run a DCF at 8% WACC" -> "Apple") is stored
unbound and attaches to the next research ticker. Switching to a different company or asking to reset
("use CAPM", "reset assumptions") drops them.
"""

import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("finance_agent.agents.query_parameters")

SESSION_KEY = "valuation_overrides"
OVERRIDE_FIELDS = ("wacc", "terminal_growth", "beta", "share_price")

WACC_BAND = (0.03, 0.25)
TERMINAL_GROWTH_BAND = (-0.02, 0.05)
BETA_BAND = (0.05, 4.0)

_NUMBER = r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
_LINK = r"\s*(?:of|is|at|=|:|to)?\s*"

_WACC_PATTERNS = (
    re.compile(rf"\b(?:wacc|discount rate|hurdle rate|cost of capital){_LINK}{_NUMBER}\s*(%?)", re.IGNORECASE),
    re.compile(rf"{_NUMBER}\s*(%)\s*(?:wacc|discount rate|hurdle rate|cost of capital)\b", re.IGNORECASE),
)
_GROWTH_PATTERNS = (
    re.compile(
        rf"\b(?:terminal|perpetual|perpetuity)\s+(?:growth\s+)?(?:rate\s+)?(?:growth\s+)?{_LINK}{_NUMBER}\s*(%?)",
        re.IGNORECASE,
    ),
    re.compile(rf"{_NUMBER}\s*(%)\s*(?:terminal|perpetual|perpetuity)\s+growth\b", re.IGNORECASE),
)
_BETA_PATTERN = re.compile(rf"\bbeta{_LINK}{_NUMBER}(?!\s*%)", re.IGNORECASE)
_PRICE_PATTERN = re.compile(
    rf"\b(?:share price|stock price|current price|price|trading at){_LINK}(\$)?\s*{_NUMBER}", re.IGNORECASE
)
_RESET_PATTERN = re.compile(
    r"\b(?:reset|clear|remove|drop)\b[^.]*\b(?:assumptions?|overrides?|inputs?|wacc|parameters?)\b|"
    r"\b(?:use|back to)\s+(?:the\s+)?(?:capm|default|live)\b",
    re.IGNORECASE,
)


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


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _rate(raw: str, percent_sign: str) -> float:
    """'8%' and '8' mean 8%; '0.08' means 8%."""
    value = _to_float(raw)
    return value / 100.0 if percent_sign or abs(value) > 1.0 else value


def _first_rate(patterns, text: str) -> Optional[float]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _rate(match.group(1), match.group(2))
    return None


def _in_band(name: str, value: Optional[float], band, notes: List[str], fmt: str) -> Optional[float]:
    if value is None:
        return None
    lo, hi = band
    if lo <= value <= hi:
        return value
    notes.append(
        f"Ignored {name} {fmt.format(value)}: outside the plausible range {fmt.format(lo)}–{fmt.format(hi)}; "
        "the model default was used."
    )
    return None


def parse_valuation_overrides(text: Optional[str]) -> ValuationOverrides:
    """Extracts range-checked valuation parameters from one user message."""
    text = text or ""
    notes: List[str] = []
    wacc = _in_band("discount rate", _first_rate(_WACC_PATTERNS, text), WACC_BAND, notes, "{:.2%}")
    growth = _in_band("terminal growth", _first_rate(_GROWTH_PATTERNS, text), TERMINAL_GROWTH_BAND, notes, "{:.2%}")

    beta_match = _BETA_PATTERN.search(text)
    beta = _in_band("beta", _to_float(beta_match.group(1)) if beta_match else None, BETA_BAND, notes, "{:.2f}")

    price = None
    for match in _PRICE_PATTERN.finditer(text):
        value = _to_float(match.group(2))
        # A bare four-digit year after "price" ("price in 2025" never matches, but "price 2025" could) is not a price.
        if not match.group(1) and value.is_integer() and 1900 <= value <= date.today().year + 10:
            continue
        if value > 0:
            price = value
            break

    if wacc is not None and growth is not None and growth >= wacc:
        notes.append(
            f"Ignored terminal growth {growth:.2%}: it must be below the {wacc:.2%} discount rate; the model default was used."
        )
        growth = None

    return ValuationOverrides(
        wacc=wacc, terminal_growth=growth, beta=beta, share_price=price,
        reset=bool(_RESET_PATTERN.search(text)), notes=notes,
    )


def resolve_session_overrides(
    current: ValuationOverrides,
    stored: Optional[Dict[str, Any]],
    ticker: Optional[str],
) -> Dict[str, Any]:
    """Merges this turn's overrides with remembered ones; returns the record to persist (may be empty).

    Current values win field by field. Remembered values apply only when they are unbound (given before a
    company was named) or bound to the same ticker; a reset request drops them.
    """
    ticker = (ticker or "").upper() or None
    carried: Dict[str, Any] = {}
    if stored and not current.reset:
        bound_to = (stored.get("ticker") or "").upper() or None
        if bound_to is None or bound_to == ticker:
            carried = {k: stored[k] for k in OVERRIDE_FIELDS if stored.get(k) is not None}
    merged = {**carried, **current.values()}
    if not merged:
        return {}
    return {"ticker": ticker, **merged}
