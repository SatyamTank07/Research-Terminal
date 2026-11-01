"""When an extracted segment or revenue table counts as reconciled, and how far the parser looks around a table."""

# Reconciliation tolerances: relative share of the reference value, plus an absolute floor in
# table units to absorb per-row rounding (each row is rounded to the table's unit).
RECONCILE_REL_TOLERANCE = 0.01
RECONCILE_ABS_UNITS_PER_ROW = 1.0
SCALE_MATCH_TOLERANCE = 0.02

MIN_SEGMENTS = 2
MIN_REVENUE_LINES = 2
GEOGRAPHIC_SHARE = 0.5
CONTEXT_TAIL_CHARS = 500
CAPTION_TAIL_CHARS = 200  # statement unit captions sit directly above the table
