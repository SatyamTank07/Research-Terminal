"""Ticker precedence for a research turn; no ticker anywhere means the desk asks which company."""

import logging
from typing import Optional

from app.agents.specialized.supervisor.modules.company_resolution.schema import ResolvedCompany

logger = logging.getLogger("finance_agent.agents.supervisor")

CLARIFY_ENTITY_MESSAGE = (
    "I am your Senior Equity Research Director. Which company or stock ticker would you like me to evaluate?"
)


def resolve_company(
    explicit_ticker: Optional[str], extracted_ticker: Optional[str], session_ticker: Optional[str]
) -> Optional[ResolvedCompany]:
    """Explicit argument, then the judge's extraction, then the active session ticker; None when all are empty."""
    if explicit_ticker:
        return ResolvedCompany(ticker=explicit_ticker, source="explicit")
    if extracted_ticker:
        return ResolvedCompany(ticker=extracted_ticker, source="extracted")
    if session_ticker:
        logger.info(f"Supervisor: no ticker extracted; continuing with the active session ticker {session_ticker}.")
        return ResolvedCompany(ticker=session_ticker, source="session")
    return None
