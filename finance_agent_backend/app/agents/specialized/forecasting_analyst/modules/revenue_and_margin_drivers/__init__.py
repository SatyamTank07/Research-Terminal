"""How fast will revenue grow, and where will margins go? Growth and margin paths and the LLM driver screen."""

from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers.calculate import (
    extend_fade,
    historical_fade_drivers,
    linear_fade,
    unsupported_margin_corridor,
)
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers.schema import (
    DriverAdjustment,
    DriverScreenResult,
    ForecastDriverProposal,
    ForecastDrivers,
)
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers.verify import screen_forecast_drivers

__all__ = [
    "DriverAdjustment",
    "DriverScreenResult",
    "ForecastDriverProposal",
    "ForecastDrivers",
    "extend_fade",
    "historical_fade_drivers",
    "linear_fade",
    "screen_forecast_drivers",
    "unsupported_margin_corridor",
]
