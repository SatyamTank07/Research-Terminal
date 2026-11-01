"""What cash does the business throw off each year? UFCF schedule, revenue CAGR, cumulative FCF and the table."""

from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule.calculate import (
    calculate_forecast_schedule,
    project_forecast_years,
    project_with_overrides,
    run_schedule,
)
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule.schema import ForecastScheduleResult, ForecastYearResult

__all__ = [
    "ForecastScheduleResult",
    "ForecastYearResult",
    "calculate_forecast_schedule",
    "project_forecast_years",
    "project_with_overrides",
    "run_schedule",
]
