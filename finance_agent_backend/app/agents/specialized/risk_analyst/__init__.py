from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.risk_analyst.agent_risk_analyst import (
    RiskAnalystAgent,
    analyze_risks,
    summarize_risks,
)
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
    "DisclosurePassage",
    "RiskAuditOutput",
    "RiskDataQuality",
    "RiskDisclosureDiff",
    "RiskEvidence",
    "RiskItem",
]
