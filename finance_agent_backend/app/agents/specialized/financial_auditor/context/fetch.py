"""Which filings the audit reads: the audit window and the Item 8 primary statements behind it."""

import logging
from typing import Any, Dict, List, Optional

from app.agents.shared.filing.table_search import retrieve_10k_tables

logger = logging.getLogger("finance_agent.agents.financial_auditor")

STATEMENT_TYPES = ("income_statement", "balance_sheet", "cash_flow")
MAX_FILINGS = 3


def resolve_audit_window(fiscal_year: int, available_fiscal_years: Optional[List[int]]) -> List[int]:
    """3 contiguous years by default; up to 5 when a filing exists for every year in the span."""
    window = [fiscal_year - 2, fiscal_year - 1, fiscal_year]
    if available_fiscal_years:
        avail = {y for y in available_fiscal_years if y <= fiscal_year}
        for candidate_min in range(fiscal_year - 4, fiscal_year - 2):
            span = list(range(candidate_min, fiscal_year + 1))
            if all(y in avail for y in span):
                return span
    return window


def fetch_statement_tables(
    ticker: str,
    fiscal_year: int,
    available_fiscal_years: Optional[List[int]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieves the primary statements for the target filing and up to two prior filings (newest first)."""
    from app.database import SessionLocal

    prior = sorted((y for y in (available_fiscal_years or []) if y < fiscal_year), reverse=True)
    filing_years = [fiscal_year] + prior[: MAX_FILINGS - 1]
    tables: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STATEMENT_TYPES}
    db = None
    try:
        db = SessionLocal()
        for fy in filing_years:
            for st in STATEMENT_TYPES:
                for res in retrieve_10k_tables(ticker, fy, statement_type=st, limit=1, db=db):
                    tables[st].append(res.model_dump())
    except Exception as e:
        logger.warning(f"Statement pre-fetch failed for {ticker} FY{fiscal_year}: {e}")
        return {}
    finally:
        if db is not None:
            db.close()
    return tables
