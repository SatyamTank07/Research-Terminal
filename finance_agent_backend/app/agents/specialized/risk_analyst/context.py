"""Item 1A for the Risk Analyst: locate it, recognise a filing that provides none, pick the prior-year filing,
note how the section was read, and pool the chunks the tool loop retrieved."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.shared.filing.sections import locate_section_chunks

logger = logging.getLogger("finance_agent.agents.risk_analyst")

RISK_SECTION_ITEM = "Item 1A"
NOT_PROVIDED_MAX_CHARS = 2_500

_NOT_PROVIDED = re.compile(
    r"not\s+applicable|smaller\s+reporting\s+compan|not\s+required\s+to\s+(?:provide|include)|\bomitted\b",
    re.IGNORECASE,
)


def section_not_provided(chunks: Sequence[NarrativeChunkResult]) -> bool:
    """True when a located section is a short statement that no disclosure is provided
    (e.g. "Not applicable" or the smaller-reporting-company exemption)."""
    text = " ".join(c.content for c in chunks).strip()
    if not text:
        return True
    return len(text) <= NOT_PROVIDED_MAX_CHARS and bool(_NOT_PROVIDED.search(text))


def resolve_prior_year(fiscal_year: int, available_fiscal_years: Optional[List[int]]) -> Optional[int]:
    """The latest filed year before `fiscal_year`; the year before when the filed years are unknown."""
    if available_fiscal_years is None:
        return fiscal_year - 1
    prior = [y for y in available_fiscal_years if y < fiscal_year]
    return max(prior) if prior else None


def locate_item1a(ticker: str, fiscal_year: int) -> Tuple[List[NarrativeChunkResult], str]:
    """Item 1A chunks and how they were found; a lookup failure reads as "not_found"."""
    try:
        return locate_section_chunks(ticker, fiscal_year, RISK_SECTION_ITEM)
    except Exception as e:
        logger.warning(f"Item 1A pre-fetch failed for {ticker} FY{fiscal_year}: {e}")
        return [], "not_found"


def coverage_notes(
    ticker: str, fiscal_year: int, coverage_mode: str, located_by: str, prior_year: Optional[int], has_diff: bool
) -> List[str]:
    """Data-quality notes on how Item 1A was read and whether the prior year could be compared."""
    notes: List[str] = []
    if coverage_mode == "rag_fallback":
        notes.append(
            f"Item 1A could not be located in the {ticker} FY{fiscal_year} 10-K (no section label or heading); "
            "risks were sourced through hybrid retrieval across the filing and may be incomplete."
        )
    if located_by == "heading_boundary":
        notes.append(
            "Item 1A was located from the filing's own section headings because the stored section labels were wrong; "
            "consider re-ingesting this filing."
        )
    elif coverage_mode == "budgeted_section":
        notes.append("Item 1A exceeded the context budget; non-summary paragraphs were truncated to their lead text.")
    if prior_year and not has_diff:
        notes.append(f"Prior-year (FY{prior_year}) Item 1A unavailable; year-over-year change not assessed.")
    return notes


def tool_chunks(messages: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    """Collects chunks (with content) returned by narrative-retrieval tool calls."""
    chunks: Dict[str, Dict[str, Any]] = {}
    for msg in messages:
        raw = getattr(msg, "content", "")
        if not isinstance(raw, str) or '"chunk_id"' not in raw or '"content"' not in raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        for item in parsed if isinstance(parsed, list) else [parsed]:
            if isinstance(item, dict) and item.get("chunk_id") and isinstance(item.get("content"), str):
                chunks.setdefault(str(item["chunk_id"]), item)
    return chunks
