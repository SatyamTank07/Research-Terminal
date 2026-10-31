from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.business_strategist.agent_business_strategist import (
    BusinessStrategistAgent,
    analyze_business,
    summarize_moat,
)
from app.agents.specialized.business_strategist.state_business_strategist import (
    BusinessMoatOutput,
    SegmentDetail,
    SegmentMoat,
    StrategistDataQuality,
    StrategyEvidence,
)

AGENT = AgentSpec(
    name="business_strategist",
    label="Business Strategist",
    output_key="business_moat",
    run=analyze_business,
    milestone=summarize_moat,
)

__all__ = [
    "AGENT",
    "BusinessStrategistAgent",
    "BusinessMoatOutput",
    "SegmentDetail",
    "SegmentMoat",
    "StrategistDataQuality",
    "StrategyEvidence",
]
