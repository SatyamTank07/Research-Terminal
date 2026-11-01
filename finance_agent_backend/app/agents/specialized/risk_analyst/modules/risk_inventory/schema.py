"""A material risk and the verbatim filing quotes behind it.

Fields marked "set by code" are derived in verify and the risk profile (quote verification, likelihood x impact
severity matrix, year-over-year change); any value the LLM supplies for them is overwritten.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

RiskCategory = Literal[
    "Operational",
    "Regulatory & Legal",
    "Supply Chain & Concentration",
    "Competitive & Demand",
    "Macroeconomic & Geopolitical",
    "Technological & Cybersecurity",
    "Financial & Capital Structure",
]
RISK_CATEGORIES = list(RiskCategory.__args__)

RatingLevel = Literal["High", "Medium", "Low"]

FinancialTransmission = Literal[
    "Revenue",
    "Gross Margin",
    "Operating Expenses",
    "Capex & Working Capital",
    "Cost of Capital",
    "Terminal Value",
    "Balance Sheet & Liquidity",
]
FINANCIAL_TRANSMISSIONS = list(FinancialTransmission.__args__)


class RiskEvidence(BaseModel):
    """A verbatim filing quote supporting a risk claim."""

    chunk_id: str = Field(..., description="Chunk the quote was copied from")
    quote: str = Field(..., description="Verbatim excerpt (<= 300 chars)")
    item: Optional[str] = Field(None, description="Filing section of the chunk (set by code)")
    breadcrumb: Optional[str] = Field(None, description="Filing location trail (set by code)")
    verified: bool = Field(False, description="True once the quote was matched against the chunk text (set by code)")


class RiskItem(BaseModel):
    """Individual material risk factor extracted from Item 1A."""

    risk_id: str = Field("", description="Rank identifier R1..Rn (set by code)")
    risk_category: RiskCategory = Field(..., description="Domain categorization of the risk")
    risk_title: str = Field(..., description="Concise headline of the specific risk")
    risk_summary: str = Field(..., description="Company-specific disclosure from Item 1A (no generic boilerplate)")
    likelihood: RatingLevel = Field(..., description="Probability the risk materializes over the forecast horizon")
    impact: RatingLevel = Field(..., description="Effect on cash flows or terminal value if realized")
    severity: Literal["Severe", "Moderate", "Low"] = Field(
        ..., description="Likelihood x impact matrix outcome (set by code)"
    )
    financial_transmission: List[FinancialTransmission] = Field(
        default_factory=list, description="Model lines the risk would hit"
    )
    quantified_exposure: Optional[str] = Field(
        None, description="Filing-stated magnitude (e.g. revenue share, charge amount); kept only if its numbers appear in the evidence"
    )
    mitigating_factors: Optional[str] = Field(
        None, description="Filing-stated mitigation; kept only if mitigation_evidence is verified"
    )
    mitigation_evidence: Optional[RiskEvidence] = Field(None, description="Verbatim quote supporting the mitigation")
    evidence: List[RiskEvidence] = Field(default_factory=list, description="Verified verbatim quotes (>= 1 required)")
    disclosure_change: Literal["New", "Expanded", "Unchanged", "Not Assessed"] = Field(
        "Not Assessed", description="Change versus the prior-year Item 1A (set by code)"
    )
    monitoring_signposts: List[str] = Field(default_factory=list, description="1-3 observable indicators to monitor")
    adjustments: List[str] = Field(default_factory=list, description="Deterministic adjustments applied (set by code)")
