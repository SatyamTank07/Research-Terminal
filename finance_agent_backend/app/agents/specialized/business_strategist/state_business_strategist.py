from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SegmentDetail(BaseModel):
    """Detailed breakdown of a primary business segment or product line (ASC 606 disaggregated revenue stream)."""

    name: str = Field(..., description="Commercial product or revenue line name (e.g. 'iPhone', 'AWS', 'Data Center')")
    description: str = Field(..., description="What the offering provides and how it monetizes")
    growth_drivers: List[str] = Field(
        default_factory=list, description="Key secular or product growth drivers disclosed in 10-K"
    )


class SegmentMoat(BaseModel):
    """Competitive economic moat evaluation for a discrete business segment or division."""

    segment_name: str = Field(..., description="Operating segment or division name")
    moat_type: Literal[
        "Network Effects",
        "Cost Advantage",
        "High Switching Costs",
        "Intangible Assets / Brand",
        "Efficient Scale",
        "None",
    ] = Field(..., description="Moat classification for this specific segment")
    moat_rationale: str = Field(
        ..., description="Evidence-backed rationale for this segment's competitive defense cited from 10-K"
    )


class BusinessMoatOutput(BaseModel):
    """Structured qualitative artifact emitted by the Business & Moat Strategist Agent."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="10-K fiscal year analyzed")

    # 1. Business Architecture & Segments (US GAAP ASC 280 vs ASC 606 Separation)
    business_summary: str = Field(
        ..., description="Concise executive synthesis of the operating model and competitive positioning"
    )
    revenue_architecture: str = Field(
        ..., description="How the company monetizes (recurring subscriptions, product sales, licensing, transactions)"
    )
    reportable_segments: List[str] = Field(
        default_factory=list,
        description="ASC 280 statutory operating segments as reported to the Chief Operating Decision Maker (CODM)",
    )
    primary_product_segments: List[str] = Field(
        default_factory=list,
        description="List of primary commercial product or disaggregated revenue lines (ASC 606 disaggregation)",
    )
    segment_details: List[SegmentDetail] = Field(
        default_factory=list,
        description="Granular breakdown of commercial product lines and growth drivers feeding Forecaster's model",
    )

    # 2. Economic Moat Evaluation (Porter / Morningstar framework)
    economic_moat_type: Literal[
        "Network Effects",
        "Cost Advantage",
        "High Switching Costs",
        "Intangible Assets / Brand",
        "Efficient Scale",
        "None",
    ] = Field(..., description="Primary overall economic moat classification")
    secondary_moat_types: List[
        Literal[
            "Network Effects",
            "Cost Advantage",
            "High Switching Costs",
            "Intangible Assets / Brand",
            "Efficient Scale",
        ]
    ] = Field(
        default_factory=list,
        description="Secondary structural competitive advantages supporting the primary moat",
    )
    segment_specific_moats: List[SegmentMoat] = Field(
        default_factory=list,
        description="Segment-specific moat deconstruction for multi-business enterprises",
    )
    moat_durability: Literal["Wide", "Narrow", "None", "Indeterminate"] = Field(
        ..., description="Durability rating (ability to defend ROIC > WACC for 10-20 years, or Indeterminate if qualitative data insufficient)"
    )
    moat_trajectory: Literal["Expanding", "Stable", "Deteriorating", "Indeterminate"] = Field(
        ..., description="Whether the competitive moat is strengthening, stable, eroding, or indeterminate"
    )
    moat_rationale: str = Field(
        ..., description="Rigorous, evidence-backed defense of moat classification cited from Item 1"
    )

    # 3. Moat Stress-Testing (Competition & Counter-Evidence)
    competitive_threats: List[str] = Field(
        default_factory=list,
        description="Material competitive pressures, industry rivals, and substitute risks cited from Item 1 Competition disclosures",
    )

    # 4. Market Power Disclosures (Strict Statutory Grounding)
    pricing_power_assessment: str = Field(
        ..., description="Evidence of pricing power vs margin vulnerability, or 'Not explicitly disclosed in 10-K'"
    )
    customer_concentration: str = Field(
        ..., description="Disclosed customer concentration (e.g. 'no single customer > 10% of sales'), or 'Not explicitly disclosed in 10-K'"
    )

    # 5. Audit Trail & Citations
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chunk IDs, breadcrumbs, and sections retrieved"
    )
