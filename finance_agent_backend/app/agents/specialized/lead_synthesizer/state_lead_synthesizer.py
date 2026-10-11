"""State and Output Schemas for Lead Synthesizer Agent."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from app.agents.tools.citation_registry import SourceRef


class ThreePillarThesis(BaseModel):
    """Institutional 3-Pillar Investment Thesis synthesized from sub-agent findings."""

    pillar_1_business_moat: str = Field(
        ..., description="Pillar 1: Business architecture, competitive moat durability, and pricing power"
    )
    pillar_2_financial_durability: str = Field(
        ..., description="Pillar 2: Earnings quality, balance sheet strength, FCF conversion, and ROIC vs WACC"
    )
    pillar_3_valuation_asymmetry: str = Field(
        ..., description="Pillar 3: Intrinsic value vs market price, margin of safety, and risk/reward asymmetry"
    )


class Final10KResearchReport(BaseModel):
    """The complete institutional equity research report delivered to the client."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    company_name: str = Field(..., description="Full legal name of the target entity")
    fiscal_year: int = Field(..., description="Target fiscal year analyzed")

    # Valuation & Stance Summary
    implied_fair_value_per_share: Optional[float] = Field(
        None, description="Base-case DCF fair value per share ($); None when the route has no valuation"
    )
    current_share_price: Optional[float] = Field(None, description="Current market share price ($)")
    upside_downside_pct: Optional[float] = Field(
        None, description="Implied upside/downside % relative to market price"
    )
    valuation_stance: Optional[Literal["Undervalued", "Fairly Valued", "Overvalued"]] = Field(
        None, description="Stance from the +/-10% margin-of-safety band; None without a valuation or market price"
    )

    # Core Institutional Narrative
    three_pillar_thesis: ThreePillarThesis = Field(
        ..., description="Structured 3-Pillar Investment Thesis"
    )
    executive_summary: str = Field(
        ..., description="Executive briefing highlighting business model, growth, and risks"
    )

    # Provenance & Audit Quality Tags
    synthesis_provenance: Literal["llm_structured", "llm_json_fallback", "template_default"] = Field(
        default="llm_structured",
        description="Quality provenance tag: llm_structured, llm_json_fallback, or template_default",
    )
    year_substituted: bool = Field(
        default=False,
        description="Discloses whether the filing year was substituted due to requested year being unavailable",
    )

    # Full Publication Report & Audit Trail
    full_markdown_report: str = Field(
        ..., description="Complete, institutional-grade Markdown publication report"
    )
    sources: List[SourceRef] = Field(
        default_factory=list,
        description="Numbered filing sources; every [[n]](#cite-n) marker in full_markdown_report resolves to sources[n-1]",
    )
