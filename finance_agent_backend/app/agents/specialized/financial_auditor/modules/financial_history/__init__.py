"""What did the last 3-5 years look like?"""

from app.agents.specialized.financial_auditor.modules.financial_history.calculate import (
    build_financial_history,
    cost_of_revenue_of,
    gross_profit_of,
    growth_pct,
    operating_expenses_of,
    pct_of,
    require_finite,
)
from app.agents.specialized.financial_auditor.modules.financial_history.schema import YearFinancialsResult

__all__ = [
    "YearFinancialsResult",
    "build_financial_history",
    "cost_of_revenue_of",
    "gross_profit_of",
    "growth_pct",
    "operating_expenses_of",
    "pct_of",
    "require_finite",
]
