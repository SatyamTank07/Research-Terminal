"""State and Output Schemas for Lead Supervisor Agent."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

QueryType = Literal[
    "full_10k_report",
    "dcf_valuation_only",
    "financial_audit_only",
    "business_moat_only",
    "risk_factors_only",
    "conversational",
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
    needs_confirmation: bool = Field(
        default=False,
        description="True if requested year is missing and requires user confirmation before proceeding",
    )
    confirmation_message: Optional[str] = Field(
        None, description="Message prompting user to confirm proceeding with the latest available fiscal year"
    )
    suggested_fiscal_year: Optional[int] = Field(
        None, description="The latest available fiscal year suggested to the user"
    )
    updated_session_state: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated active session state to persist to PostgreSQL conversation record"
    )
