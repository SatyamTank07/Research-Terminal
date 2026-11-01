"""Does the forecast hang together? Growth vs reinvestment support, implied ROIC and RONIC warnings."""

from app.agents.specialized.forecasting_analyst.modules.forecast_consistency.calculate import compute_diagnostics
from app.agents.specialized.forecasting_analyst.modules.forecast_consistency.schema import ForecastDiagnostics

__all__ = ["ForecastDiagnostics", "compute_diagnostics"]
