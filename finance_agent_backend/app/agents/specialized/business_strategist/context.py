"""Outside data the Strategist reads: Item 1 (Business) of the 10-K, located by its label or the filing's own
headings, with a search fallback, and the whole filing in order."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.agents.shared.filing.narrative_search import NarrativeChunkResult, retrieve_10k_narrative
from app.agents.shared.filing.sections import build_section_context, locate_section_chunks
from app.agents.tools.business_disclosure_tools import FilingChunk, infer_section_items, load_filing_chunks

logger = logging.getLogger("finance_agent.agents.business_strategist")

ITEM1_SECTION = "Item 1"
ITEM1_CHAR_BUDGET = 90_000
RAG_FALLBACK_LIMIT = 6
RAG_FALLBACK_QUERIES = (
    "business overview products services customers how the company generates revenue",
    "competition competitive factors competitors market position",
    "intellectual property brand network scale technology advantages",
)

Pool = Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class Item1Text:
    """Item 1 as the judge reads it: the prompt context, the quotable chunks (pool) and how it was found."""

    pool: Pool
    context: str
    coverage_mode: str
    item1_chunks: int
    chars_read: int
    notes: Tuple[str, ...]


def locate_item1(ticker: str, fiscal_year: int) -> Tuple[List[NarrativeChunkResult], str]:
    """Item 1 chunks in filing order and how they were found; ([], "not_found") when the lookup fails."""
    try:
        return locate_section_chunks(ticker, fiscal_year, ITEM1_SECTION)
    except Exception as e:
        logger.warning(f"Item 1 pre-fetch failed for {ticker} FY{fiscal_year}: {e}")
        return [], "not_found"


def load_filing(ticker: str, fiscal_year: int) -> Tuple[List[FilingChunk], List[str]]:
    """Every chunk of the 10-K in filing order, plus a note when the filing could not be loaded."""
    try:
        return load_filing_chunks(ticker, fiscal_year), []
    except Exception as e:
        logger.warning(f"Filing load failed for {ticker} FY{fiscal_year}: {e}")
        return [], ["Filing tables could not be loaded; segment and revenue-line figures not extracted."]


def filing_items(filing: List[FilingChunk], located_by: str) -> Dict[str, str]:
    """chunk_id -> true 10-K item from the filing's headings, only when the stored labels proved wrong."""
    return infer_section_items(filing) if located_by == "heading_boundary" else {}


def search_business_passages(ticker: str, fiscal_year: int) -> List[NarrativeChunkResult]:
    """Hybrid retrieval across the filing when Item 1 has neither a label nor a heading."""
    found: Dict[str, NarrativeChunkResult] = {}
    for query in RAG_FALLBACK_QUERIES:
        try:
            for chunk in retrieve_10k_narrative(ticker, fiscal_year, query, section_item=None, limit=RAG_FALLBACK_LIMIT):
                found.setdefault(chunk.chunk_id, chunk)
        except Exception as e:
            logger.warning(f"Item 1 retrieval fallback query failed for {ticker} FY{fiscal_year}: {e}")
    return sorted(found.values(), key=lambda c: c.chunk_index)


def read_item1(ticker: str, fiscal_year: int, item1: List[NarrativeChunkResult], located_by: str) -> Item1Text:
    """Builds the judge's Item 1 context under the character budget, searching the filing when Item 1 was not found."""
    if not item1:
        item1 = search_business_passages(ticker, fiscal_year)
    pool: Pool = {c.chunk_id: c.model_dump() for c in item1}
    if item1:
        context, mode, chars = build_section_context(item1, ITEM1_CHAR_BUDGET)
        coverage_mode = "rag_fallback" if located_by == "not_found" else mode
    else:
        context, chars = "", 0
        coverage_mode = "no_context"
    return Item1Text(
        pool=pool,
        context=context,
        coverage_mode=coverage_mode,
        item1_chunks=len(item1) if located_by != "not_found" else 0,
        chars_read=chars,
        notes=tuple(coverage_notes(located_by, coverage_mode, ticker, fiscal_year)),
    )


def coverage_notes(located_by: str, coverage_mode: str, ticker: str, fiscal_year: int) -> List[str]:
    notes: List[str] = []
    if located_by == "heading_boundary":
        notes.append(
            "Item 1 was located from the filing's own section headings because the stored section labels were wrong; "
            "citations use heading-inferred items."
        )
    if coverage_mode == "rag_fallback":
        notes.append(
            f"Item 1 could not be located in the {ticker} FY{fiscal_year} 10-K; business context was retrieved by "
            "search across the filing and may be incomplete."
        )
    elif coverage_mode == "budgeted_section":
        notes.append("Item 1 exceeded the context budget; later paragraphs were truncated to their lead text.")
    elif coverage_mode == "no_context":
        notes.append(f"No Item 1 text could be retrieved for {ticker} FY{fiscal_year}.")
    return notes


def pool_chunk(chunk: FilingChunk, ticker: str, fiscal_year: int, items: Dict[str, str]) -> Dict[str, Any]:
    """A filing chunk in the pool's shape, carrying its true item when headings were inferred."""
    return {
        "chunk_id": chunk.chunk_id,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "item": items.get(chunk.chunk_id) or chunk.item,
        "breadcrumb": chunk.breadcrumb,
        "sub_section": chunk.sub_section,
        "content": chunk.content,
        "chunk_index": chunk.chunk_index,
    }
