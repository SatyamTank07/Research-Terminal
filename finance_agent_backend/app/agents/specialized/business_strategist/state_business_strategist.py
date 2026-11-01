from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.agents.specialized.business_strategist.evidence import StrategyEvidence
from app.agents.specialized.business_strategist.modules.business_model.schema import SegmentDetail
from app.agents.specialized.business_strategist.modules.revenue_mix.schema import RevenueLineFigure, SegmentFigure

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
NOT_DISCLOSED = "Not explicitly disclosed in 10-K"


class SegmentMoat(BaseModel):
    """Competitive economic moat evaluation for a discrete business segment or division."""

    segment_name: str = Field(..., description="Operating segment or division name")
    moat_type: MoatType = Field(..., description="Moat classification for this specific segment")
    moat_rationale: str = Field(
        ..., description="Evidence-backed rationale for this segment's competitive defense cited from 10-K"
    )


class StrategistDataQuality(BaseModel):
    """How much of the filing was read, what the code extracted, and what was rejected."""

    coverage_mode: Literal["full_section", "budgeted_section", "rag_fallback", "no_context"] = "no_context"
    section_located_by: Literal["item_label", "heading_boundary", "not_found"] = "not_found"
    item1_chunks: int = 0
    chars_read: int = 0
    supplementary_passages: int = Field(0, description="Concentration / pricing sentences surfaced from outside Item 1")
    segment_tables_found: int = 0
    segment_layout: Optional[str] = None
    segment_basis: Literal["business", "geographic", "mixed", "single_segment", "not_found"] = "not_found"
    segment_reconciliation: Literal["in_table_total", "income_statement", "unreconciled", "not_applicable"] = "not_applicable"
    revenue_tables_found: int = 0
    revenue_lines_reconciliation: Literal["in_table_total", "income_statement", "not_applicable"] = "not_applicable"
    consolidated_revenue: Optional[float] = Field(None, description="Latest consolidated revenue (USD millions)")
    units: str = "USD millions"
    llm_status: Literal["ok", "failed", "malformed", "skipped"] = "skipped"
    claims_proposed: int = 0
    claims_verified: int = 0
    claims_dropped: int = 0
    quotes_verified: int = 0
    quotes_rejected: int = 0
    quotes_off_topic: int = Field(0, description="Verified quotes that do not evidence the claim's mechanism")
    notes: List[str] = Field(default_factory=list)


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
        description="ASC 280 reportable segments, taken from the code-extracted segment note",
    )
    primary_product_segments: List[str] = Field(
        default_factory=list,
        description="Top-level product / service revenue lines (ASC 606 disaggregation table when one exists)",
    )
    segment_details: List[SegmentDetail] = Field(
        default_factory=list,
        description="Granular breakdown of commercial product lines and growth drivers feeding Forecaster's model",
    )
    segment_financials: List[SegmentFigure] = Field(
        default_factory=list,
        description="Code-extracted revenue and segment profit by reportable segment and fiscal year (USD millions)",
    )
    revenue_lines: List[RevenueLineFigure] = Field(
        default_factory=list,
        description="Code-extracted net sales by product / service line and fiscal year (USD millions)",
    )
    segment_table_markdown: str = Field(
        "", description="Verified latest-year segment table (revenue, share, profit, margin) to embed verbatim"
    )
    revenue_mix_markdown: str = Field(
        "", description="Verified latest-year revenue-mix table by product / service line to embed verbatim"
    )

    # 2. Economic Moat Evaluation (Porter / Morningstar framework)
    economic_moat_type: MoatType = Field(..., description="Primary overall economic moat classification")
    secondary_moat_types: List[SecondaryMoatType] = Field(
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
    moat_assessment: Literal["verified", "no_moat", "unsupported", "not_assessed"] = Field(
        "not_assessed",
        description="verified: backed by filing quotes; no_moat: the model found none; unsupported: claim lacked a "
        "verified quote and was downgraded; not_assessed: no qualitative analysis ran",
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
    evidence: List[StrategyEvidence] = Field(default_factory=list, description="Verified quotes behind each claim")
    data_quality: StrategistDataQuality = Field(default_factory=StrategistDataQuality)
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chunk IDs, breadcrumbs, and sections retrieved"
    )
