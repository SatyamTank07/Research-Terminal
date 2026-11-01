"""What return do investors require? CAPM cost of equity, cost of debt, capital-structure weights."""

from app.agents.specialized.valuation_specialist.modules.cost_of_capital.calculate import (
    blume_adjust,
    calculate_wacc,
    estimate_cost_of_capital,
)
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.gather import gather_capital_inputs
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import (
    CostOfCapital,
    WACCCalculationResult,
)

__all__ = [
    "CostOfCapital",
    "WACCCalculationResult",
    "blume_adjust",
    "calculate_wacc",
    "estimate_cost_of_capital",
    "gather_capital_inputs",
]
