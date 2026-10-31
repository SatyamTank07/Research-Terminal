from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.lead_synthesizer.agent_lead_synthesizer import (
    LeadSynthesizerAgent,
    summarize_report,
    write_research_report,
)
from app.agents.specialized.lead_synthesizer.state_lead_synthesizer import (
    Final10KResearchReport,
    ThreePillarThesis,
)

AGENT = AgentSpec(
    name="lead_synthesizer",
    label="Lead Synthesizer",
    output_key="final_report",
    run=write_research_report,
    milestone=summarize_report,
    uses=frozenset({"business_moat", "financial_audit", "forecast", "dcf_valuation", "risk_audit"}),
    needs_any=frozenset({"business_moat", "financial_audit", "risk_audit"}),
    writes_final=True,
)

__all__ = [
    "AGENT",
    "LeadSynthesizerAgent",
    "Final10KResearchReport",
    "ThreePillarThesis",
]
