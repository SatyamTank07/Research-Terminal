"""Which agents run for each report type, in order; a nested list runs in parallel and joins before the next step."""

from typing import Dict, List, Sequence, Union

from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.business_strategist import AGENT as business_strategist
from app.agents.specialized.financial_auditor import AGENT as financial_auditor
from app.agents.specialized.forecasting_analyst import AGENT as forecasting_analyst
from app.agents.specialized.lead_synthesizer import AGENT as lead_synthesizer
from app.agents.specialized.risk_analyst import AGENT as risk_analyst
from app.agents.specialized.valuation_specialist import AGENT as valuation_specialist

Step = Union[AgentSpec, Sequence[AgentSpec]]

CONVERSATIONAL = "conversational"  # answered by the Supervisor; never routed to agents

ROUTES: Dict[str, List[Step]] = {
    "full_10k_report": [
        [business_strategist, financial_auditor, risk_analyst],
        forecasting_analyst,
        valuation_specialist,
        lead_synthesizer,
    ],
    "dcf_valuation_only": [financial_auditor, forecasting_analyst, valuation_specialist, lead_synthesizer],
    "financial_audit_only": [financial_auditor, lead_synthesizer],
    "business_moat_only": [business_strategist, lead_synthesizer],
    "risk_factors_only": [risk_analyst, lead_synthesizer],
}


def step_specs(step: Step) -> List[AgentSpec]:
    return [step] if isinstance(step, AgentSpec) else list(step)


def all_specs(routes: Dict[str, List[Step]] = ROUTES) -> List[AgentSpec]:
    """Every routed agent once, in first-seen order."""
    seen: Dict[str, AgentSpec] = {}
    for steps in routes.values():
        for step in steps:
            for spec in step_specs(step):
                seen.setdefault(spec.name, spec)
    return list(seen.values())
