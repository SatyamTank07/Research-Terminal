from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.risk_analyst.agent_risk_analyst import (
    RiskAnalystAgent,
    analyze_risks,
    build_risk_audit,
    summarize_risks,
)
from app.agents.specialized.risk_analyst.modules.quantified_exposures import QuantifiedDisclosure
from app.agents.specialized.risk_analyst.state_risk_analyst import (
    DisclosurePassage,
    RiskAuditOutput,
    RiskDataQuality,
    RiskDisclosureDiff,
    RiskEvidence,
    RiskItem,
)

AGENT = AgentSpec(
    name="risk_analyst",
    label="Risk Analyst",
    output_key="risk_audit",
    run=analyze_risks,
    milestone=summarize_risks,
)

__all__ = [
    "AGENT",
    "RiskAnalystAgent",
    "build_risk_audit",
    "DisclosurePassage",
    "QuantifiedDisclosure",
    "RiskAuditOutput",
    "RiskDataQuality",
    "RiskDisclosureDiff",
    "RiskEvidence",
    "RiskItem",
]
