"""State and Output Schemas for Lead Supervisor Agent."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

QueryType = Literal[
    "full_10k_report",
    "dcf_valuation_only",
    "financial_audit_only",
    "business_moat_only",
    "risk_factors_only",
]


class RoutingPlan(BaseModel):
    """Execution blueprint generated deterministically by the Lead Supervisor."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    company_name: Optional[str] = Field(None, description="Company corporate name")
    fiscal_year: int = Field(..., description="Resolved target 10-K fiscal year")
    year_requested: Optional[int] = Field(
        None, description="Original fiscal year requested by user if explicitly specified"
    )
    year_substituted: bool = Field(
        default=False,
        description="Explicit provenance flag: True if requested year was unavailable and catalog substituted latest year",
    )
    document_id: Optional[str] = Field(None, description="UUID of document in documents table")
    query_type: QueryType = Field(..., description="Execution path for the pipeline")
    active_agents: List[str] = Field(
        ..., description="Ordered list of agent identifiers required to fulfill the query"
    )
    routing_provenance: Literal["deterministic_rule", "llm_inferred"] = Field(
        default="deterministic_rule",
        description="Provenance tag: whether intent was resolved by deterministic rules or LLM inference",
    )
