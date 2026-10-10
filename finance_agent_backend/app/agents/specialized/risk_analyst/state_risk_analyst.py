"""State and Output Schemas for Risk & Red Flag Analyst Agent.

Fields marked "set by code" are derived deterministically in the agent's post-processing
(quote verification, likelihood x impact severity matrix, year-over-year diff); any value the
LLM supplies for them is overwritten.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.agents.tools.risk_disclosure_tools import DisclosurePassage, RiskDisclosureDiff

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


class RiskDataQuality(BaseModel):
    """Coverage and verification diagnostics for the risk audit."""

    coverage_mode: Literal["full_section", "budgeted_section", "rag_fallback", "not_provided"] = "rag_fallback"
    section_located_by: Literal["item_label", "heading_boundary", "not_found"] = Field(
        "not_found", description="How Item 1A was found: stored label, heading boundaries (labels were wrong), or not at all"
    )
    item1a_chunks: int = 0
    chars_read: int = 0
    prior_fiscal_year: Optional[int] = None
    risks_proposed: int = 0
    risks_dropped_unverified: int = 0
    risks_merged: int = 0
    quotes_rejected: int = 0
    quantified_disclosures: int = Field(0, description="Item 1A sentences stating a figure (code-extracted checklist)")
    quantified_uncovered: List[str] = Field(
        default_factory=list, description="Checklist sentences not cited by any verified risk"
    )
    notes: List[str] = Field(default_factory=list)


class RiskAuditOutput(BaseModel):
    """Structured qualitative artifact emitted by the Risk & Red Flag Analyst Agent."""

    ticker: str = Field(..., description="Stock ticker symbol")
    fiscal_year: int = Field(..., description="10-K fiscal year analyzed")

    # 1. Material Risk Inventory (5 to 8 items, ranked by severity, impact, likelihood)
    identified_risks: List[RiskItem] = Field(
        default_factory=list, description="Curated, evidence-verified material risks ranked by severity"
    )

    # 2. Structural / Existential Assessment
    primary_threat_risk_id: Optional[str] = Field(
        None, description="risk_id of the single biggest structural threat (re-pointed by code if invalid)"
    )
    primary_existential_threat: str = Field(
        "", description="Why the primary threat could impair terminal value"
    )
    overall_risk_profile: Literal["High", "Moderate", "Low", "Not Assessed"] = Field(
        "Not Assessed", description="Aggregate rating derived from the severity mix; Not Assessed when no verified risk exists (set by code)"
    )

    # 3. Year-over-year disclosure change
    disclosure_changes: Optional[RiskDisclosureDiff] = None

    # 4. Deterministic artifacts & audit trail
    data_quality: RiskDataQuality = Field(default_factory=RiskDataQuality)
    risk_matrix_markdown: str = Field("", description="Verified risk matrix table (set by code)")
    evidence_table_markdown: str = Field("", description="Verified evidence table (set by code)")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chunks backing verified evidence, with their true item and breadcrumb"
    )


__all__ = [
    "RISK_CATEGORIES",
    "FINANCIAL_TRANSMISSIONS",
    "DisclosurePassage",
    "RiskDisclosureDiff",
    "RiskEvidence",
    "RiskItem",
    "RiskDataQuality",
    "RiskAuditOutput",
]
