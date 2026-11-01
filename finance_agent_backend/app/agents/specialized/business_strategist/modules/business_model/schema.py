"""A product line or segment as the business model describes it."""

from typing import List

from pydantic import BaseModel, Field


class SegmentDetail(BaseModel):
    """Detailed breakdown of a primary business segment or product line (ASC 606 disaggregated revenue stream)."""

    name: str = Field(..., description="Commercial product or revenue line name (e.g. 'iPhone', 'AWS', 'Data Center')")
    description: str = Field(..., description="What the offering provides and how it monetizes")
    growth_drivers: List[str] = Field(
        default_factory=list, description="Key secular or product growth drivers disclosed in 10-K"
    )
