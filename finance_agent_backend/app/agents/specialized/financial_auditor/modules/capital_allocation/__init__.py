"""Does management earn more than the capital costs, and where does the cash go?"""

from app.agents.specialized.financial_auditor.modules.capital_allocation.calculate import (
    measure_capital_intensity,
    measure_capital_returns,
    measure_share_count,
    screen_capital_allocation,
)
from app.agents.specialized.financial_auditor.modules.capital_allocation.schema import (
    CapitalReturns,
    YearCapitalIntensity,
    YearShareCount,
)

__all__ = [
    "CapitalReturns",
    "YearCapitalIntensity",
    "YearShareCount",
    "measure_capital_intensity",
    "measure_capital_returns",
    "measure_share_count",
    "screen_capital_allocation",
]
