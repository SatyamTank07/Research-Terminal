"""Solver ranges and tolerances for what the share price implies."""

# Reverse DCF: constant explicit-period revenue growth that makes the base DCF equal the share price, with
# margins, tax, the CapEx/D&A/working-capital logic, WACC, terminal growth, terminal RONIC and the equity
# bridge held fixed. Scanned on a grid, then bisected inside the sign-change bracket nearest the modeled
# CAGR: bounded and always terminates. The range spans steep decline to hyper-growth for 5-10 years.
REVERSE_DCF_GROWTH_BOUNDS = (-0.20, 0.60)
# Companion solve: uniform operating-margin shift (decimal pp) at modeled growth, margins kept in [-200%, 100%].
REVERSE_DCF_MARGIN_SHIFT_BOUNDS = (-0.30, 0.50)
REVERSE_DCF_SCAN_POINTS = 33
REVERSE_DCF_MAX_ITERATIONS = 60
REVERSE_DCF_TOLERANCE = 1e-5
REVERSE_DCF_GAP_WARNING_PP = 5.0
IMPLIED_WACC_UPPER = 0.30
IMPLIED_WACC_MIN_SPREAD_OVER_G = 0.005
