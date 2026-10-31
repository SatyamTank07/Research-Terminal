from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.valuation_specialist.agent_valuation_specialist import (
    ValuationSpecialistAgent,
    summarize_valuation,
    value_equity,
)
from app.agents.specialized.valuation_specialist.state_valuation_specialist import (
    DCFValuationOutput,
    WACCAudit,
)

AGENT = AgentSpec(
    name="valuation_specialist",
    label="Valuation Specialist",
    output_key="dcf_valuation",
    run=value_equity,
    milestone=summarize_valuation,
    needs=frozenset({"financial_audit"}),
    uses=frozenset({"forecast"}),
    skip_message="Valuation skipped: no audited financial statements were available.",
)

__all__ = [
    "AGENT",
    "ValuationSpecialistAgent",
    "DCFValuationOutput",
    "WACCAudit",
]
