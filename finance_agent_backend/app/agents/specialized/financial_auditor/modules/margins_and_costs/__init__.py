"""How profitable is each dollar of revenue, and is that changing?"""

from app.agents.specialized.financial_auditor.modules.margins_and_costs.calculate import (
    measure_cost_structure,
    screen_margins_and_costs,
)
from app.agents.specialized.financial_auditor.modules.margins_and_costs.schema import YearCostStructure

__all__ = [
    "YearCostStructure",
    "measure_cost_structure",
    "screen_margins_and_costs",
]
