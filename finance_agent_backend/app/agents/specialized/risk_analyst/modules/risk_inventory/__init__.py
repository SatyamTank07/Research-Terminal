"""What are the material risks, how likely, how damaging, and does the filing back each claim?"""

from app.agents.specialized.risk_analyst.modules.risk_inventory.verify import (
    is_boilerplate,
    states_immaterial,
    substantiates_high_impact,
)

__all__ = [
    "is_boilerplate",
    "states_immaterial",
    "substantiates_high_impact",
]
