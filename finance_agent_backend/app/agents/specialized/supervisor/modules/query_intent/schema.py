"""The judge's answer about the user's request, and the verified intent the checklist acts on."""

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field

QueryType = Literal[
    "full_10k_report",
    "dcf_valuation_only",
    "financial_audit_only",
    "business_moat_only",
    "risk_factors_only",
    "conversational",
]


class SupervisorExtraction(BaseModel):
    """Pydantic schema for supervisor intent and entity extraction."""

    query_type: QueryType = Field(
        default="conversational",
        description="Selected routing path (full_10k_report, dcf_valuation_only, financial_audit_only, business_moat_only, risk_factors_only, conversational)",
    )
    extracted_ticker: Optional[str] = Field(
        None, description="Extracted stock ticker if identifiable (e.g. AAPL, TSLA, NVDA)"
    )
    extracted_year: Optional[int] = Field(
        None, description="Extracted 4-digit fiscal year if explicitly requested by user (e.g. 2023)"
    )
    conversational_response: Optional[str] = Field(
        None,
        description=(
            "If query_type == 'conversational', provide the full, professional response "
            "directly to the user as a Senior Equity Research Director. If query requires 10-K execution, leave null."
        ),
    )
    routing_reasoning: str = Field(
        default="", description="Brief explanation of the routing classification"
    )


@dataclass(frozen=True)
class QueryIntent:
    """The report type to run, whether the LLM or the fallback rule chose it, and the reply for a chat turn."""

    query_type: str
    provenance: Literal["deterministic_rule", "llm_inferred"]
    reply: Optional[str]
