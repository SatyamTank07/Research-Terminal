"""What if things go better or worse? Bull / base / bear schedules and their 25 / 50 / 25 weights."""

from app.agents.specialized.forecasting_analyst.modules.scenarios.calculate import build_scenarios
from app.agents.specialized.forecasting_analyst.modules.scenarios.schema import ForecastScenario
from app.agents.specialized.forecasting_analyst.modules.scenarios.thresholds import SCENARIO_PROBABILITIES

__all__ = ["ForecastScenario", "SCENARIO_PROBABILITIES", "build_scenarios"]
