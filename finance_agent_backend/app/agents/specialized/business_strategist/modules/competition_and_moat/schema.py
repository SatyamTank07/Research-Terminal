"""The moat types a filing can evidence, and a moat held by one segment."""

from typing import List, Literal

from pydantic import BaseModel, Field

MoatType = Literal[
    "Network Effects",
    "Cost Advantage",
    "High Switching Costs",
    "Intangible Assets / Brand",
    "Efficient Scale",
    "None",
]
SecondaryMoatType = Literal[
    "Network Effects",
    "Cost Advantage",
    "High Switching Costs",
    "Intangible Assets / Brand",
    "Efficient Scale",
]
MOAT_TYPES: List[str] = list(MoatType.__args__)


class SegmentMoat(BaseModel):
    """Competitive economic moat evaluation for a discrete business segment or division."""

    segment_name: str = Field(..., description="Operating segment or division name")
    moat_type: MoatType = Field(..., description="Moat classification for this specific segment")
    moat_rationale: str = Field(
        ..., description="Evidence-backed rationale for this segment's competitive defense cited from 10-K"
    )
