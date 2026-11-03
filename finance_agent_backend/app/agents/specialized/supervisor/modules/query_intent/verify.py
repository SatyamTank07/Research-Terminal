"""Screens the judge's intent: an unknown report type or a failed call becomes a conversational turn."""

from typing import Optional, get_args

from app.agents.specialized.supervisor.modules.query_intent.schema import QueryIntent, QueryType, SupervisorExtraction

CONVERSATIONAL = "conversational"
DEFAULT_REPLY = (
    "I am your Senior Equity Research Director. I can assist you with comprehensive SEC 10-K research reports, "
    "three-statement financial audits, multi-year UFCF projections, and DCF valuations. "
    "Which company or ticker would you like to evaluate?"
)


def verify_query_intent(extraction: Optional[SupervisorExtraction]) -> QueryIntent:
    """Keeps a known report type; a conversational turn always carries a reply."""
    if extraction and extraction.query_type in get_args(QueryType):
        query_type, provenance = extraction.query_type, "llm_inferred"
    else:
        query_type, provenance = CONVERSATIONAL, "deterministic_rule"
    if query_type != CONVERSATIONAL:
        return QueryIntent(query_type=query_type, provenance=provenance, reply=None)
    reply = extraction.conversational_response if (extraction and extraction.conversational_response) else DEFAULT_REPLY
    return QueryIntent(query_type=query_type, provenance=provenance, reply=reply)
