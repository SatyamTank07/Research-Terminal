"""Cut-offs for intrinsic value: DCF input bounds, terminal-value sanity bands, the RONIC cap and the placeholder forecast."""

TV_CONCENTRATION_WARNING_PCT = 85.0
STALE_ROLL_FORWARD_YEARS = 1.5

# Plausibility bands for the multiples implied by the terminal value (TV at year N / year-N metric).
# EV/EBITDA: developed-market companies rarely sustain below ~5x outside distress or cyclical peaks, and a
# perpetuity growing at <= 3.5% cannot justify more than ~30x. EV/NOPAT is the cleaner check because the
# value-driver identity gives TV/NOPAT = (1 + g)(1 - g/RONIC)/(WACC - g): about 7x at a 15% WACC with no
# value-creating growth, and only above ~35x with a very low WACC and a very high RONIC. A multiple outside
# these bands usually means the final-year reinvestment or margin is not a steady state.
TERMINAL_EV_EBITDA_BAND = (5.0, 30.0)
TERMINAL_EV_NOPAT_BAND = (6.0, 35.0)

MAX_WACC_RATE = 0.50
MAX_TERMINAL_GROWTH_RATE = 0.05
MIN_TERMINAL_GROWTH_RATE = -0.02
TERMINAL_GROWTH_WARNING_RATE = 0.035

RONIC_CAP = 0.30
NAIVE_FALLBACK_GROWTH = 0.05
NAIVE_FALLBACK_YEARS = 5
DAYS_PER_YEAR = 365.25
