"""Relabels the section metadata of already-ingested SEC filings without re-embedding.

Re-derives `part`, `item`, `sub_section`, `breadcrumb`, the metadata JSON section keys and the
breadcrumb-dependent `content_tsv` for every chunk of the selected filings (see
app/services/ingestion/section_relabel.py). Chunk text and embeddings are never modified, and only
chunks whose labels change are written, so the script is idempotent.

Usage (inside the backend container):
    python scripts/relabel_sections.py --ticker AMZN --dry-run
    python scripts/relabel_sections.py --ticker AMZN --fiscal-year 2025
    python scripts/relabel_sections.py --all --dry-run
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import func  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.models import Document, DocumentChunk  # noqa: E402
from app.services.ingestion.section_relabel import RelabelPlan, StoredChunk, format_plan, plan_relabel  # noqa: E402
from app.services.ingestion.sec_parser import SECFilingMetadata  # noqa: E402

logger = logging.getLogger("finance_agent.scripts.relabel_sections")


def _select_documents(db: Session, ticker: Optional[str], fiscal_year: Optional[int], filing_type: Optional[str]) -> List[Document]:
    query = db.query(Document)
    if ticker:
        query = query.filter(Document.ticker == ticker.upper())
    if fiscal_year:
        query = query.filter(Document.fiscal_year == fiscal_year)
    if filing_type:
        query = query.filter(Document.filing_type == filing_type)
    return query.order_by(Document.ticker, Document.filing_type, Document.fiscal_year).all()


def plan_document(db: Session, doc: Document) -> RelabelPlan:
    metadata = SECFilingMetadata(
        ticker=doc.ticker,
        company_name=doc.company_name,
        filing_type=doc.filing_type,
        fiscal_year=doc.fiscal_year,
        period_end_date=doc.period_end_date,
        source_filename=doc.source_filename,
    )
    rows = []
    for chunk in db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all():
        meta = chunk.metadata_ or {}
        section_title = meta.get("section_title") or (chunk.sub_section if chunk.chunk_type == "narrative" else None)
        stored = StoredChunk(
            chunk_index=chunk.chunk_index,
            chunk_type=chunk.chunk_type,
            content=chunk.content,
            part=chunk.part,
            item=chunk.item,
            section_title=section_title,
            table_title=meta.get("table_title"),
        )
        rows.append((chunk.id, stored, chunk.breadcrumb))
    return plan_relabel(metadata, rows)


def apply_plan(db: Session, plan: RelabelPlan) -> None:
    """Writes the planned labels in one transaction (caller commits or rolls back)."""
    by_id = {change.chunk_id: change for change in plan.changes}
    if not by_id:
        return
    for chunk in db.query(DocumentChunk).filter(DocumentChunk.id.in_(list(by_id))).all():
        after = by_id[chunk.id].after
        chunk.part = after.part
        chunk.item = after.item
        chunk.sub_section = after.sub_section
        chunk.breadcrumb = after.breadcrumb
        chunk.metadata_ = {**(chunk.metadata_ or {}), "part": after.part, "item": after.item, "section_title": after.section_title}
        chunk.content_tsv = func.to_tsvector("english", after.breadcrumb + " " + chunk.content)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--ticker", help="Relabel this ticker's filings")
    scope.add_argument("--all", action="store_true", help="Relabel every ingested filing")
    parser.add_argument("--fiscal-year", type=int, help="Limit to one fiscal year")
    parser.add_argument("--filing-type", help="Limit to one form type, e.g. 10-K")
    parser.add_argument("--dry-run", action="store_true", help="Print before/after item counts without writing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    db = SessionLocal()
    failures = 0
    try:
        documents = _select_documents(db, args.ticker, args.fiscal_year, args.filing_type)
        if not documents:
            logger.error("No ingested filings match the selection.")
            return 1
        print(f"{'DRY RUN - no changes written' if args.dry_run else 'APPLYING'}: {len(documents)} filing(s)\n")
        for doc in documents:
            try:
                plan = plan_document(db, doc)
                print(format_plan(plan))
                if not args.dry_run and plan.changes:
                    apply_plan(db, plan)
                    db.commit()
                    print(f"    committed {len(plan.changes)} chunk updates")
            except Exception:
                db.rollback()
                failures += 1
                logger.exception(f"Relabel failed for {doc.ticker} {doc.filing_type} FY{doc.fiscal_year}; no changes kept for it.")
            print()
    finally:
        db.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
