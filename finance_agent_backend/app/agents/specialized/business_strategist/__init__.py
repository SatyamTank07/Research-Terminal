from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.business_strategist.agent_business_strategist import (
    BusinessStrategistAgent,
    analyze_business,
    summarize_moat,
)
from app.agents.specialized.business_strategist.context import FilingChunk
from app.agents.specialized.business_strategist.evidence import StrategyEvidence
from app.agents.specialized.business_strategist.modules.business_model import SegmentDetail
from app.agents.specialized.business_strategist.modules.competition_and_moat import SegmentMoat
from app.agents.specialized.business_strategist.modules.revenue_mix import RevenueLineFigure, SegmentFigure
from app.agents.specialized.business_strategist.state_business_strategist import (
    BusinessMoatOutput,
    StrategistDataQuality,
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
    "FilingChunk",
    "RevenueLineFigure",
    "SegmentDetail",
    "SegmentFigure",
    "SegmentMoat",
    "StrategistDataQuality",
    "StrategyEvidence",
]
