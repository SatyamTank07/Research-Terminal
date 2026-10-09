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
    available_fiscal_years: List[int] = Field(
        default_factory=list,
        description="All distinct fiscal years available in the catalog for this ticker, sorted DESC",
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
    conversational_response: Optional[str] = Field(
        default=None,
        description="Direct professional conversational reply generated immediately when query_type == 'conversational'",
    )
    needs_confirmation: bool = Field(
        default=False,
        description="Deprecated: Retained for backward compatibility. Automatic year substitution is used instead.",
    )
    confirmation_message: Optional[str] = Field(
        None, description="Deprecated: Retained for backward compatibility."
    )
    suggested_fiscal_year: Optional[int] = Field(
        None, description="Deprecated: Retained for backward compatibility."
    )
    updated_session_state: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated active session state to persist to PostgreSQL conversation record"
    )
