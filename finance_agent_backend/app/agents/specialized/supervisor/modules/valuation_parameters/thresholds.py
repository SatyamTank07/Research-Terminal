"""Plausible ranges for user-specified valuation inputs, and the house terminal growth."""

WACC_BAND = (0.03, 0.25)
TERMINAL_GROWTH_BAND = (-0.02, 0.05)
BETA_BAND = (0.05, 4.0)

DEFAULT_TERMINAL_GROWTH = 0.025  # house perpetual growth when the user gives none
