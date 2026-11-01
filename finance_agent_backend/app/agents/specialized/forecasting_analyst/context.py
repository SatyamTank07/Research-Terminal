"""Outside data the forecast needs: the Item 7 (MD&A) excerpts that may carry management's outlook."""

import logging
from typing import List

from app.agents.shared.filing.narrative_search import NarrativeChunkResult, retrieve_10k_narrative

logger = logging.getLogger("finance_agent.agents.forecasting_analyst")

MDNA_QUERY = "outlook expect demand growth pricing margins costs capital expenditures guidance"
MDNA_CHUNK_LIMIT = 6


def fetch_mdna_chunks(ticker: str, fiscal_year: int) -> List[NarrativeChunkResult]:
    """Top Item 7 excerpts for the outlook query; an empty list when retrieval fails (the forecast then fades)."""
    try:
        return retrieve_10k_narrative(
            ticker=ticker, fiscal_year=fiscal_year, query=MDNA_QUERY,
            section_item="Item 7", limit=MDNA_CHUNK_LIMIT,
        )
    except Exception as e:
        logger.warning(f"MD&A retrieval failed for {ticker} FY{fiscal_year}: {e}")
        return []
