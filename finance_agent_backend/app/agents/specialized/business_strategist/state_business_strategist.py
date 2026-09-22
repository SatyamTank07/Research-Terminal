from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

class SegmentDetail(BaseModel):
    """Detailed breakdown of a primary business segment or product line."""

    name: str = Field(..., description="Segment or product line name (e.g. 'iPhone', 'Services')")
    description: str = Field(..., description="What the segment offers and how it monetizes")
    growth_drivers: List[str] = Field(
        default_factory=list, description="Key secular or product growth drivers disclosed in 10-K"
    )


class BusinessMoatOutput(BaseModel):
    """Structured qualitative artifact emitted by the Business & Moat Strategist Agent."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="10-K fiscal year analyzed")

    # 1. Business Architecture & Segments
    business_summary: str = Field(
        ..., description="Concise executive synthesis of the operating model and competitive positioning"
    )
    revenue_architecture: str = Field(
        ..., description="How the company monetizes (hardware, recurring SaaS/services, licenses, transactions)"
    )
    primary_product_segments: List[str] = Field(
        ..., description="List of primary operating or reporting segments (e.g. ['iPhone', 'Mac', 'Services'])"
    )
    segment_details: List[SegmentDetail] = Field(
        default_factory=list,
        description="Granular segment breakdown feeding directly into Forecaster's per-segment growth model",
    )

    # 2. Economic Moat Evaluation (Porter / Morningstar framework)
    economic_moat_type: Literal[
        "Network Effects",
        "Cost Advantage",
        "High Switching Costs",
        "Intangible Assets / Brand",
        "Efficient Scale",
        "None",
    ] = Field(..., description="Primary economic moat classification")
    moat_durability: Literal["Wide", "Narrow", "None"] = Field(
        ..., description="Durability rating (ability to defend ROIC > WACC for 10-20 years)"
    )
    moat_trajectory: Literal["Expanding", "Stable", "Deteriorating"] = Field(
        ..., description="Whether the competitive moat is strengthening, stable, or eroding"
    )
    moat_rationale: str = Field(
        ..., description="Rigorous, evidence-backed defense of moat classification cited from Item 1"
    )

    # 3. Market Power Disclosures
    pricing_power_assessment: str = Field(
        ..., description="Evidence of pricing power vs margin vulnerability from Item 1"
    )
    customer_concentration: str = Field(
        ..., description="Disclosed customer concentration (e.g. 'no single customer > 10% of sales')"
    )

    # 4. Audit Trail
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chunk IDs, breadcrumbs, and sections retrieved"
    )

