"""Locating a 10-K section's chunks (stored item label, else the filing's own headings) and rendering
them as `[chunk:<id>] text` prompt context under a character budget."""

import re
from typing import Any, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from app.agents.shared.filing.narrative_search import NarrativeChunkResult

DEFAULT_CHAR_BUDGET = 160_000
BUDGETED_CHUNK_LEAD_CHARS = 600
MAX_SUMMARY_CHUNKS = 6


def _to_result(chunk: Any, doc: Any, item: Optional[str] = None) -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=str(chunk.id),
        document_id=str(doc.id),
        ticker=doc.ticker,
        fiscal_year=doc.fiscal_year,
        item=item or chunk.item or "Narrative",
        breadcrumb=chunk.breadcrumb,
        sub_section=chunk.sub_section,
        content=chunk.content,
        chunk_index=chunk.chunk_index,
    )


def _filing_chunks_query(db: Session, ticker: str, fiscal_year: int):
    from app.models import ANNUAL_REPORT_FILING_TYPE, Document, DocumentChunk

    return (
        db.query(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(
            Document.ticker == ticker.strip().upper(),
            Document.fiscal_year == fiscal_year,
            Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
        )
    )


def fetch_section_chunks(
    ticker: str,
    fiscal_year: int,
    item: str,
    db: Optional[Session] = None,
) -> List[NarrativeChunkResult]:
    """Returns every narrative chunk stored under `item` for the 10-K, in filing order (exact item match)."""
    from app.database import SessionLocal
    from app.models import DocumentChunk

    owns_db = db is None
    if owns_db:
        db = SessionLocal()
    try:
        rows = (
            _filing_chunks_query(db, ticker, fiscal_year)
            .filter(DocumentChunk.chunk_type == "narrative", DocumentChunk.item == item)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        return [_to_result(chunk, doc) for chunk, doc in rows]
    finally:
        if owns_db:
            db.close()


# Standard Form 10-K item order and title keywords, used to find section boundaries from the
# headings themselves when the stored item labels are wrong (e.g. headings rendered as tables).
TEN_K_ITEM_ORDER = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
    "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
]
TEN_K_ITEM_TITLES = {
    "1": r"business", "1A": r"risk\s+factors", "1B": r"unresolved\s+staff", "1C": r"cybersecurity",
    "2": r"properties", "3": r"legal\s+proceedings", "4": r"mine\s+safety", "5": r"market\s+for",
    "6": r"reserved|selected", "7": r"management", "7A": r"quantitative\s+and\s+qualitative",
    "8": r"financial\s+statements", "9": r"changes\s+in\s+and\s+disagreements", "9A": r"controls\s+and\s+procedures",
    "9B": r"other\s+information", "9C": r"foreign\s+jurisdictions", "10": r"directors", "11": r"executive\s+compensation",
    "12": r"security\s+ownership", "13": r"certain\s+relationships", "14": r"principal\s+account",
    "15": r"exhibits?", "16": r"form\s+10-k\s+summary",
}
HEADING_MAX_CHARS = 300
_HEADING_LEAD = re.compile(r"^[\s|#*>_:\-]*item\s*(\d{1,2}[a-c]?)\b\.?[\s|:\-—*]*", re.IGNORECASE)
_ITEM_MENTION = re.compile(r"\bitem\s*(\d{1,2}[a-c]?)\b", re.IGNORECASE)


def heading_item(content: str) -> Optional[str]:
    """Returns the 10-K item a chunk opens with ("1A", "7"...) if the chunk is a section heading.

    Short chunks (e.g. a heading rendered as a one-row table) qualify on the "Item N" lead alone;
    longer chunks must also carry the item's title right after it, so cross-references such as
    "see Item 1A" do not start a section. Tables of contents (>= 3 distinct items) never qualify.
    """
    match = _HEADING_LEAD.match(content or "")
    if not match:
        return None
    number = match.group(1).upper()
    if number not in TEN_K_ITEM_TITLES:
        return None
    if len({m.upper() for m in _ITEM_MENTION.findall(content)}) >= 3:
        return None
    if len(content) <= HEADING_MAX_CHARS:
        return number
    following = content[match.end():match.end() + 120]
    return number if re.match(rf"[\W_]*(?:{TEN_K_ITEM_TITLES[number]})", following, re.IGNORECASE) else None


def locate_section_chunks(
    ticker: str,
    fiscal_year: int,
    item: str,
    db: Optional[Session] = None,
) -> Tuple[List[NarrativeChunkResult], str]:
    """Finds a 10-K section's narrative chunks. Returns (chunks, located_by).

    located_by:
    - "item_label": chunks stored under the exact item label.
    - "heading_boundary": labels were wrong, so the section was cut from the filing between its own
      heading and the next later item heading; chunks are relabeled with `item`.
    - "not_found": no label and no heading for the item.
    """
    from app.database import SessionLocal
    from app.models import DocumentChunk

    target = item.split()[-1].upper()
    owns_db = db is None
    if owns_db:
        db = SessionLocal()
    try:
        labeled = fetch_section_chunks(ticker, fiscal_year, item, db=db)
        if labeled:
            return labeled, "item_label"
        if target not in TEN_K_ITEM_ORDER:
            return [], "not_found"

        rows = _filing_chunks_query(db, ticker, fiscal_year).order_by(DocumentChunk.chunk_index.asc()).all()
        target_rank = TEN_K_ITEM_ORDER.index(target)
        start = next((i for i, (chunk, _) in enumerate(rows) if heading_item(chunk.content) == target), None)
        if start is None:
            return [], "not_found"

        section: List[NarrativeChunkResult] = []
        start_chunk, start_doc = rows[start]
        if start_chunk.chunk_type == "narrative" and len(start_chunk.content) > HEADING_MAX_CHARS:
            section.append(_to_result(start_chunk, start_doc, item))
        for chunk, doc in rows[start + 1:]:
            found = heading_item(chunk.content)
            if found is not None and TEN_K_ITEM_ORDER.index(found) > target_rank:
                break
            if chunk.chunk_type == "narrative":
                section.append(_to_result(chunk, doc, item))
        return section, "heading_boundary"
    finally:
        if owns_db:
            db.close()


def _is_summary_chunk(content: str) -> bool:
    return content.count("•") >= 3 or "summary" in content[:400].lower()


def build_section_context(
    chunks: Sequence[NarrativeChunkResult],
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> Tuple[str, str, int]:
    """Renders chunks as `[chunk:<id>] text` blocks. Returns (context, coverage_mode, chars_included).

    Within budget every chunk is included in full ("full_section"). Over budget the leading
    "Summary of Risk Factors" chunks stay whole and every other chunk is cut to its lead text,
    which carries the risk heading ("budgeted_section").
    """
    blocks = [(c.chunk_id, c.content.strip()) for c in chunks]
    total = sum(len(text) for _, text in blocks)
    if total <= char_budget:
        mode = "full_section"
        parts = [f"[chunk:{cid}] {text}" for cid, text in blocks]
    else:
        mode = "budgeted_section"
        parts = []
        used = 0
        leading_summary = True
        for idx, (cid, text) in enumerate(blocks):
            if leading_summary and idx < MAX_SUMMARY_CHUNKS and _is_summary_chunk(text):
                body = text
            else:
                leading_summary = False
                body = text[:BUDGETED_CHUNK_LEAD_CHARS] + (" [...]" if len(text) > BUDGETED_CHUNK_LEAD_CHARS else "")
            if used + len(body) > char_budget:
                break
            parts.append(f"[chunk:{cid}] {body}")
            used += len(body)
    context = "\n\n".join(parts)
    return context, mode, sum(len(p) for p in parts)


__all__ = [
    "DEFAULT_CHAR_BUDGET",
    "HEADING_MAX_CHARS",
    "TEN_K_ITEM_ORDER",
    "TEN_K_ITEM_TITLES",
    "build_section_context",
    "fetch_section_chunks",
    "heading_item",
    "locate_section_chunks",
]
