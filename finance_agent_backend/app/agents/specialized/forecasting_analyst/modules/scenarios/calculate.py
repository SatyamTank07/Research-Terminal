"""What if things go better or worse? Bull and bear cases shift growth and margin around the base case."""

from typing import Any, Dict, List, Optional

from app.agents.specialized.forecasting_analyst.modules.audited_baseline import MARGIN_FLOOR, ForecastBaseline
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule import run_schedule
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path import apply_reinvestment_path
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import ForecastDrivers
from app.agents.specialized.forecasting_analyst.modules.scenarios.schema import ForecastScenario
from app.agents.specialized.forecasting_analyst.modules.scenarios.thresholds import (
    BEAR_MARGIN_SHIFT,
    BULL_MARGIN_SHIFT,
    MAX_SCENARIO_GROWTH_SHIFT,
    MIN_SCENARIO_GROWTH_SHIFT,
    RISK_MULTIPLIERS,
    SCENARIO_ORDER,
    SCENARIO_PROBABILITIES,
)


def build_scenarios(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    base_schedule: Dict[str, Any],
    risk_profile: Optional[str],
    terminal_growth: float,
) -> List[ForecastScenario]:
    """Growth shifts are largest in year 1 and vanish by the final year, so every case meets terminal growth.

    Bull and bear are defined by free cash flow: when the base case burns cash, faster growth burns more,
    so the growth shift flips sign while the margin shift keeps its direction. Each case rebuilds its own
    CapEx steady state from its final-year NOPAT, matching how the valuation sizes its terminal value.
    """
    shift = min(max(baseline.growth_volatility or 0.0, MIN_SCENARIO_GROWTH_SHIFT), MAX_SCENARIO_GROWTH_SHIFT)
    if base_schedule["cumulative_fcf"] < 0:
        shift = -shift
    risk_mult = RISK_MULTIPLIERS.get(risk_profile or "", 1.0)
    n = len(drivers.revenue_growth)
    weights = [1.0] if n == 1 else [(n - 1 - i) / (n - 1) for i in range(n)]
    variants = {
        "bull": (shift, BULL_MARGIN_SHIFT),
        "bear": (-shift * risk_mult, BEAR_MARGIN_SHIFT * risk_mult),
    }
    scenarios = [_scenario("base", drivers, base_schedule)]
    for name, (g_shift, m_shift) in variants.items():
        variant, _ = apply_reinvestment_path(baseline, drivers.model_copy(update={
            "revenue_growth": [round(g + g_shift * w, 4) for g, w in zip(drivers.revenue_growth, weights)],
            "operating_margin": [round(min(max(m + m_shift, MARGIN_FLOOR), 1.0), 4) for m in drivers.operating_margin],
        }), terminal_growth)
        scenarios.append(_scenario(name, variant, run_schedule(baseline, variant)))
    return sorted(scenarios, key=lambda s: SCENARIO_ORDER[s.name])


def _scenario(name: str, drivers: ForecastDrivers, schedule: Dict[str, Any]) -> ForecastScenario:
    return ForecastScenario(
        name=name,
        probability=SCENARIO_PROBABILITIES[name],
        revenue_growth_pct=[round(g * 100.0, 2) for g in drivers.revenue_growth],
        operating_margin_pct=[round(m * 100.0, 2) for m in drivers.operating_margin],
        projected_fcfs=schedule["projected_fcfs"],
        revenue_cagr_pct=schedule["revenue_cagr_pct"],
        cumulative_fcf=schedule["cumulative_fcf"],
    )
