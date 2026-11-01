"""Bull / base / bear shifts and weights (decimals)."""

SCENARIO_PROBABILITIES = {"bull": 0.25, "base": 0.50, "bear": 0.25}
SCENARIO_ORDER = {"bull": 0, "base": 1, "bear": 2}
MIN_SCENARIO_GROWTH_SHIFT = 0.02
MAX_SCENARIO_GROWTH_SHIFT = 0.10
BULL_MARGIN_SHIFT = 0.01
BEAR_MARGIN_SHIFT = -0.015
RISK_MULTIPLIERS = {"High": 1.5, "Moderate": 1.0, "Low": 0.75}
