"""What is it worth if the bull or bear case plays out? Each forecast scenario is valued with the base case's
WACC, terminal growth and equity bridge; the base scenario reuses the headline DCF."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from app.agents.specialized.forecasting_analyst import SCENARIO_PROBABILITIES
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import CostOfCapital
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.calculate import calculate_dcf
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.schema import IntrinsicValue
from app.agents.specialized.valuation_specialist.modules.scenario_values.render import render_scenario_table
from app.agents.specialized.valuation_specialist.modules.scenario_values.schema import ScenarioValuation, ScenarioValues
from app.agents.specialized.valuation_specialist.upside import upside_pct

if TYPE_CHECKING:
    from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput


def _scenario_terminal_nopat(forecast: "ForecastOutput", scenario, tax: float) -> Optional[float]:
    """Final-year NOPAT of a scenario rebuilt from its growth and margin path (same formula as the schedule)."""
    if not scenario.revenue_growth_pct or not scenario.operating_margin_pct:
        return None
    revenue = forecast.base_revenue
    for g in scenario.revenue_growth_pct:
        revenue *= 1.0 + g / 100.0
    return revenue * scenario.operating_margin_pct[-1] / 100.0 * (1.0 - tax)


def value_scenarios(
    forecast: Optional["ForecastOutput"], intrinsic: IntrinsicValue, capital: CostOfCapital, price: Optional[float]
) -> ScenarioValues:
    """Values each forecast scenario on the base case's terms. Scenario DCF warnings are not reported: the
    headline DCF already carries the sanity checks."""
    valued: List[ScenarioValuation] = []
    if forecast is not None:
        for sc in forecast.scenarios:
            if sc.name == "base":
                ev, per_share = intrinsic.dcf.enterprise_value, intrinsic.dcf.fair_value_per_share
            else:
                res = calculate_dcf(
                    projected_fcfs=sc.projected_fcfs,
                    terminal_nopat=_scenario_terminal_nopat(forecast, sc, capital.marginal_tax_rate),
                    **intrinsic.terms.as_kwargs(),
                )
                ev, per_share = res.enterprise_value, res.fair_value_per_share
            valued.append(ScenarioValuation(
                name=sc.name,
                probability=SCENARIO_PROBABILITIES.get(sc.name, sc.probability),
                revenue_cagr_pct=sc.revenue_cagr_pct,
                cumulative_fcf=sc.cumulative_fcf,
                enterprise_value=ev,
                fair_value_per_share=per_share,
                upside_downside_pct=upside_pct(per_share, price),
            ))
    by_name = {s.name: s for s in valued}
    weighted = (
        round(sum(s.probability * s.fair_value_per_share for s in valued), 2)
        if {"bull", "base", "bear"} <= set(by_name) else None
    )
    table = ""
    if valued:
        table = render_scenario_table(valued, weighted, upside_pct(weighted, price) if weighted is not None else None)
    return ScenarioValues(
        scenario_valuations=valued,
        probability_weighted_fair_value=weighted,
        bear_fair_value=by_name["bear"].fair_value_per_share if "bear" in by_name else None,
        bull_fair_value=by_name["bull"].fair_value_per_share if "bull" in by_name else None,
        scenario_table_markdown=table,
    )
