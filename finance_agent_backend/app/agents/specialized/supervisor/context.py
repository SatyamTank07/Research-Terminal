"""What the Supervisor reads before it decides: the active session and the ingested 10-K catalog."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.database import SessionLocal
from app.models import ANNUAL_REPORT_FILING_TYPE, Document

CATALOG_HINT_LIMIT = 25


class FilingCatalogError(ValueError):
    """No usable 10-K for the request; the message tells the user which tickers are ingested."""


@dataclass(frozen=True)
class FilingRecord:
    """One ingested 10-K."""

    ticker: str
    company_name: str
    fiscal_year: int
    document_id: str


def active_session(session_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The conversation's research context, without the transient `pending_action` key."""
    return {k: v for k, v in (session_state or {}).items() if k != "pending_action"}


def _catalog_hint(db, limit: int = CATALOG_HINT_LIMIT) -> str:
    """Short list of ingested tickers (capped so a large catalog never floods the reply)."""
    tickers = [
        row[0]
        for row in db.query(Document.ticker)
        .filter(Document.filing_type == ANNUAL_REPORT_FILING_TYPE)
        .distinct()
        .order_by(Document.ticker)
        .limit(limit + 1)
        .all()
    ]
    if not tickers:
        return "No 10-K filings have been ingested yet; upload one to begin."
    shown = ", ".join(tickers[:limit])
    more = " and others" if len(tickers) > limit else ""
    return f"Ingested 10-K tickers include: {shown}{more}."


def load_filing_catalog(ticker: Optional[str]) -> List[int]:
    """Fiscal years with an ingested 10-K for the ticker, newest first; raises when there are none."""
    resolved_ticker = (ticker or "").strip().upper()
    db = SessionLocal()
    try:
        if not resolved_ticker:
            raise FilingCatalogError(
                f"Could not identify a target stock ticker from the request. {_catalog_hint(db)}"
            )
        year_records = (
            db.query(Document.fiscal_year)
            .filter(Document.ticker == resolved_ticker, Document.filing_type == ANNUAL_REPORT_FILING_TYPE)
            .distinct()
            .order_by(Document.fiscal_year.desc())
            .all()
        )
        available_years = [r[0] for r in year_records]
        if not available_years:
            raise FilingCatalogError(
                f"No ingested 10-K filings found for ticker '{resolved_ticker}'. {_catalog_hint(db)}"
            )
        return available_years
    finally:
        db.close()


def load_filing(ticker: str, fiscal_year: int) -> FilingRecord:
    """The ticker's 10-K for the fiscal year, else its latest one."""
    resolved_ticker = (ticker or "").strip().upper()
    db = SessionLocal()
    try:
        query = db.query(Document).filter(
            Document.ticker == resolved_ticker, Document.filing_type == ANNUAL_REPORT_FILING_TYPE
        )
        doc = query.filter(Document.fiscal_year == fiscal_year).first()
        if not doc:
            doc = query.order_by(Document.fiscal_year.desc()).first()
        return FilingRecord(
            ticker=doc.ticker, company_name=doc.company_name, fiscal_year=doc.fiscal_year, document_id=doc.id
        )
    finally:
        db.close()
