from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.forecasting_analyst.agent_forecasting_analyst import (
    ForecastingAnalystAgent,
    forecast_cash_flows,
    summarize_forecast,
)
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import (
    ForecastOutput,
    ForecastYear,
    GuidanceSource,
)
from app.agents.tools.forecast_baseline import MARGIN_FLOOR
from app.agents.tools.forecast_model import SCENARIO_PROBABILITIES, project_with_overrides

AGENT = AgentSpec(
    name="forecasting_analyst",
    label="Forecasting Analyst",
    output_key="forecast",
    run=forecast_cash_flows,
    milestone=summarize_forecast,
    needs=frozenset({"financial_audit"}),
    uses=frozenset({"business_moat", "risk_audit"}),
    skip_message="Forecast skipped: no audited financial statements were available.",
)

__all__ = [
    "AGENT",
    "ForecastingAnalystAgent",
    "ForecastOutput",
    "ForecastYear",
    "GuidanceSource",
    "MARGIN_FLOOR",
    "SCENARIO_PROBABILITIES",
    "project_with_overrides",
]
