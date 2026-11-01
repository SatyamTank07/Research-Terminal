"""Upside of a fair value against the share price, shared by the scenario and stance modules."""

from typing import Optional


def upside_pct(fair_value: float, price: Optional[float]) -> Optional[float]:
    """Fair value against the share price in percent (positive = upside); None without a positive price."""
    if price is None or price <= 0:
        return None
    return round((fair_value - price) / price * 100.0, 2)
