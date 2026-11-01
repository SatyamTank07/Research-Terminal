"""What is the audited starting point? Base revenue and margin, tax rate, reinvestment intensities and RONIC."""

from app.agents.specialized.forecasting_analyst.modules.audited_baseline.calculate import build_forecast_baseline
from app.agents.specialized.forecasting_analyst.modules.audited_baseline.render import render_history_table
from app.agents.specialized.forecasting_analyst.modules.audited_baseline.schema import ForecastBaseline, HistoricalRow
from app.agents.specialized.forecasting_analyst.modules.audited_baseline.thresholds import MARGIN_FLOOR

__all__ = ["ForecastBaseline", "HistoricalRow", "MARGIN_FLOOR", "build_forecast_baseline", "render_history_table"]
