"""Code-owned bounds of the audited baseline (ratios are decimals unless the name says _pct)."""

CAPEX_AVERAGE_YEARS = 3
CAPEX_RECENCY_WEIGHTS = (1.0, 2.0, 3.0)  # oldest -> latest over the last CAPEX_AVERAGE_YEARS audited years
STEADY_STATE_RONIC_FLOOR = 0.08
STEADY_STATE_RONIC_CAP = 0.30
STEADY_STATE_RONIC_DEFAULT = 0.10
MARGIN_TREND_YEARS = 3
MARGIN_TREND_MIN_CHANGE = 0.01
MAX_PLAUSIBLE_TAX_RATE_PCT = 50.0
MARGIN_FLOOR = -2.0
NWC_INTENSITY_BOUND = 0.5
