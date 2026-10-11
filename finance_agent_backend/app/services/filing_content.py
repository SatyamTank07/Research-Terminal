"""Read-side helpers for the filing viewer: a whole 10-K rebuilt from its chunks, and re-ingestion lookup."""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models import ANNUAL_REPORT_FILING_TYPE, Document, DocumentChunk


def _item_title(item: str, breadcrumb: Optional[str]) -> str:
    """The breadcrumb segment naming the item (e.g. 'Item 7: MD&A'), else the bare item label."""
    for segment in (breadcrumb or "").split(" > "):
        if segment.strip().lower().startswith(item.lower()):
            return segment.strip()
    return item


def get_filing_content(db: Session, document_id: str) -> Optional[Dict[str, Any]]:
    """Every chunk of a filing in reading order plus an item table of contents; None when the filing is unknown."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        return None
    rows = (
        db.query(
            DocumentChunk.id, DocumentChunk.chunk_index, DocumentChunk.part, DocumentChunk.item,
            DocumentChunk.breadcrumb, DocumentChunk.chunk_type, DocumentChunk.content,
        )
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    toc: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        if r.item and (r.part, r.item) not in seen:
            seen.add((r.part, r.item))
            toc.append({"part": r.part, "item": r.item, "title": _item_title(r.item, r.breadcrumb), "chunk_index": r.chunk_index})
    return {
        "document": document.to_dict(),
        "toc": toc,
        "chunks": [
            {
                "id": r.id, "chunk_index": r.chunk_index, "part": r.part, "item": r.item,
                "breadcrumb": r.breadcrumb, "chunk_type": r.chunk_type, "content": r.content,
            }
            for r in rows
        ],
    }


def locate_chunk(
    db: Session,
    ticker: str,
    fiscal_year: int,
    breadcrumb: Optional[str] = None,
    item: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Closest chunk in the current 10-K for ticker/year: same breadcrumb, else same item, else nearest index."""
    document = (
        db.query(Document)
        .filter(
            Document.ticker == ticker.upper(),
            Document.fiscal_year == fiscal_year,
            Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
        )
        .order_by(Document.created_at.desc())
        .first()
    )
    if document is None:
        return None
    rows = (
        db.query(DocumentChunk.id, DocumentChunk.chunk_index, DocumentChunk.item, DocumentChunk.breadcrumb)
        .filter(DocumentChunk.document_id == document.id)
        .all()
    )
    if not rows:
        return None
    target = chunk_index if chunk_index is not None else 0
    pool = (
        [r for r in rows if breadcrumb and r.breadcrumb == breadcrumb]
        or [r for r in rows if item and r.item == item]
        or rows
    )
    best = min(pool, key=lambda r: (abs(r.chunk_index - target), r.chunk_index))
    return {"document_id": document.id, "chunk_id": best.id, "exact": False}
