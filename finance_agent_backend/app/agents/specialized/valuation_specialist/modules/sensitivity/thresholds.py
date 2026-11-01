"""Steps of the sensitivity grids around the base case (decimals) and the RONIC-grid guards."""

# Sensitivity grid steps
SENSITIVITY_WACC_STEPS = (-0.010, -0.005, 0.0, 0.005, 0.010)
SENSITIVITY_GROWTH_STEPS = (-0.006, -0.003, 0.0, 0.003, 0.006)
SENSITIVITY_RONIC_STEPS = (-0.04, -0.02, 0.0, 0.05, 0.10)
RONIC_GRID_MIN_SPREAD_OVER_G = 0.01
RONIC_EQUALS_WACC_TOLERANCE = 0.005
