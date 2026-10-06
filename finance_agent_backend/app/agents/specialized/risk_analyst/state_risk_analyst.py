"""State and Output Schemas for Risk & Red Flag Analyst Agent."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class RiskItem(BaseModel):
    """Individual material risk factor extracted from Item 1A."""

    risk_category: Literal[
        "Operational",
        "Regulatory & Legal",
        "Supply Chain & Concentration",
        "Macroeconomic & Geopolitical",
        "Technological & Cybersecurity",
    ] = Field(..., description="Domain categorization of the risk")
    risk_title: str = Field(..., description="Concise headline of the specific risk")
    risk_summary: str = Field(
        ..., description="Specific company disclosure from Item 1A (avoiding generic boilerplate)"
    )
    severity: Literal["Severe", "Moderate", "Low"] = Field(
        ..., description="Estimated impact on cash flows or terminal value if realized"
    )
    mitigating_factors: Optional[str] = Field(
        None, description="Disclosed management offsets or hedging strategies, if stated in the filing"
    )


class RiskAuditOutput(BaseModel):
    """Structured qualitative artifact emitted by the Risk & Red Flag Analyst Agent."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="10-K fiscal year analyzed")

    # 1. Material Risk Inventory (5 to 8 items, sorted by severity)
    identified_risks: List[RiskItem] = Field(
        ..., description="Curated list of 5-8 material, non-boilerplate risks sorted by severity"
    )

    # 2. Structural / Existential Assessment
    primary_existential_threat: str = Field(
        ..., description="The single biggest structural threat disclosed in Item 1A that could impair terminal value"
    )
    overall_risk_profile: Literal["High", "Moderate", "Conservative"] = Field(
        ..., description="One-line aggregate qualitative risk rating used directly by the Lead Synthesizer"
    )

    # 3. Audit Trail
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chunk IDs, breadcrumbs, and sections retrieved"
    )
