"""Catalog and Filing Discovery Tools.

Provides scalable PostgreSQL search across ingested SEC filings (10-K, 10-Q),
enabling dynamic entity and fiscal year resolution without dumping the entire
database catalog into LLM system prompts (O(1) prompt context scaling).
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from app.database import SessionLocal
from app.models import Document

logger = logging.getLogger("finance_agent.tools.catalog")


@tool
def search_sec_catalog(query: Optional[str] = "") -> List[Dict[str, Any]]:
    """Searches the PostgreSQL document catalog for matching stock tickers, company names, and available fiscal years.

    Use this tool whenever a user mentions a company name or ticker to verify availability before routing,
    or to discover which fiscal periods are currently ingested in the SEC filing repository.

    Args:
        query: Company name (e.g. 'Apple', 'Tesla') or stock ticker symbol (e.g. 'AAPL', 'TSLA').

    Returns:
        List of matching document metadata records with ticker, company_name, fiscal_year, document_id, and filing_type.
    """
    clean_q = (query or "").strip()
    if not clean_q:
        return []

    db = SessionLocal()
    try:
        results = (
            db.query(Document.ticker, Document.company_name, Document.fiscal_year, Document.id, Document.filing_type)
            .filter(
                (Document.ticker.ilike(f"{clean_q}%"))
                | (Document.company_name.ilike(f"%{clean_q}%"))
            )
            .order_by(Document.ticker, Document.fiscal_year.desc())
            .limit(10)
            .all()
        )
        return [
            {
                "ticker": r[0],
                "company_name": r[1],
                "fiscal_year": r[2],
                "document_id": str(r[3]),
                "filing_type": r[4],
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Error querying SEC document catalog for '{query}': {e}")
        return []
    finally:
        db.close()
