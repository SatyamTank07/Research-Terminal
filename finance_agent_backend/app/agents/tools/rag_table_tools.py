"""Universal 10-K RAG Table Retrieval Tool.

Enables quantitative agents (Financial Auditor, Forecaster, Valuation Specialist)
to deterministically retrieve exact, multi-year Markdown financial tables from
ingested SEC 10-K filings without vector embedding distortion or arithmetic hallucination.

Uses Universal US-GAAP / IFRS accounting taxonomy to ensure robust retrieval across
any public company filing without hardcoding company-specific products or divisions.
"""

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, not_
from langchain_core.tools import tool

from app.database import SessionLocal
from app.models import Document, DocumentChunk

logger = logging.getLogger("finance_agent.tools.rag_tables")

# ==============================================================================
# 1. Universal US-GAAP / IFRS Accounting Signatures
# ==============================================================================
# Standard financial line items used across all public 10-K filings.
UNIVERSAL_STATEMENT_SIGNATURES: Dict[str, List[str]] = {
    "income_statement": [
        "total net sales",
        "total revenues",
        "operating revenues",
        "net sales",
        "total revenue",
        "operating income",
        "income from operations",
        "costs of sales",
    ],
    "balance_sheet": [
        "current assets",
        "total assets",
        "cash and cash equivalents",
        "total liabilities",
        "shareholders’ equity",
        "stockholders’ equity",
        "total liabilities and stockholders' equity",
        "total liabilities and shareholders' equity",
    ],
    "cash_flow": [
        "cash flows from operating activities",
        "operating activities",
        "cash provided by operating activities",
        "net cash provided by operating activities",
        "investing activities",
        "financing activities",
    ],
    "segments": [
        "segment",
        "reportable segment",
        "disaggregated revenue",
        "geographic information",
        "revenue by reportable segment",
    ],
}

# Normalize common caller aliases
STATEMENT_TYPE_ALIASES: Dict[str, str] = {
    "income": "income_statement",
    "income_statement": "income_statement",
    "operations": "income_statement",
    "statement_of_operations": "income_statement",
    "statements_of_operations": "income_statement",
    "balance": "balance_sheet",
    "balance_sheet": "balance_sheet",
    "balance_sheets": "balance_sheet",
    "consolidated_balance_sheets": "balance_sheet",
    "cash_flow": "cash_flow",
    "cash_flows": "cash_flow",
    "statement_of_cash_flows": "cash_flow",
    "statements_of_cash_flows": "cash_flow",
    "segment": "segments",
    "segments": "segments",
    "segment_reporting": "segments",
}

# Non-financial index/TOC table markers to filter out
INDEX_TABLE_PATTERNS = [
    "%index to consolidated financial statements%",
    "%index to financial statements%",
]


# ==============================================================================
# 2. Structured Output Data Schema
# ==============================================================================
class TableChunkResult(BaseModel):
    """Structured container for a retrieved 10-K financial table with source citations."""
    chunk_id: str = Field(..., description="Unique database UUID of the chunk for citations")
    document_id: str = Field(..., description="Parent 10-K document ID")
    ticker: str = Field(..., description="Stock symbol (e.g. AAPL, TSLA)")
    fiscal_year: int = Field(..., description="Fiscal year of the filing")
    item: str = Field(..., description="SEC 10-K Item (e.g. Item 8, Item 7)")
    breadcrumb: str = Field(..., description="Filing location trail")
    sub_section: Optional[str] = Field(None, description="Section or table title")
    table_markdown: str = Field(..., description="Pristine Markdown representation of the table")
    statement_type: Optional[str] = Field(None, description="Classified standard statement type")
    row_count: int = Field(0, description="Number of table rows")
    column_count: int = Field(0, description="Number of table columns")


# ==============================================================================
# 3. Core Deterministic Retrieval Functions
# ==============================================================================
def retrieve_10k_tables(
    ticker: str,
    fiscal_year: int,
    statement_type: Optional[str] = None,
    keyword_filter: Optional[str] = None,
    section_item: Optional[str] = "Item 8",
    limit: int = 5,
    db: Optional[Session] = None,
) -> List[TableChunkResult]:
    """
    Directly retrieves exact multi-year Markdown tables from a target 10-K filing.
    
    Filters strictly by DocumentChunk.chunk_type == 'table', bypassing vector search.
    Preserves filing reading order (chunk_index ASC) so primary financial statements
    appear before detailed footnote schedules. Excludes Table of Contents / Index tables.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA').
        fiscal_year: Target 10-K fiscal year (e.g. 2025).
        statement_type: Optional standard statement filter ('income_statement', 
                        'balance_sheet', 'cash_flow', 'segments').
        keyword_filter: Optional arbitrary text keyword (e.g. 'debt', 'leases', 'goodwill').
        section_item: Target 10-K section (defaults to 'Item 8' for financial statements).
        limit: Maximum number of tables to return (default 5).
        db: Optional existing SQLAlchemy session; if None, creates and closes its own.

    Returns:
        List of TableChunkResult models containing clean markdown grids and citation data.
    """
    owns_db = False
    if db is None:
        db = SessionLocal()
        owns_db = True

    try:
        query = (
            db.query(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .filter(
                Document.ticker == ticker.strip().upper(),
                Document.fiscal_year == fiscal_year,
                DocumentChunk.chunk_type == "table",
            )
        )

        # Normalize statement type and GAAP signatures
        normalized_statement: Optional[str] = None
        signatures = None
        sig_clauses = []
        if statement_type:
            raw_key = statement_type.strip().lower().replace(" ", "_").replace("-", "_")
            normalized_statement = STATEMENT_TYPE_ALIASES.get(raw_key, raw_key)
            signatures = UNIVERSAL_STATEMENT_SIGNATURES.get(normalized_statement)
            if signatures:
                sig_clauses = [
                    DocumentChunk.content.ilike(f"%{sig}%") for sig in signatures
                ]

        def _execute_query(target_sec: Optional[str]):
            q = (
                db.query(DocumentChunk, Document)
                .join(Document, Document.id == DocumentChunk.document_id)
                .filter(
                    Document.ticker == ticker.strip().upper(),
                    Document.fiscal_year == fiscal_year,
                    DocumentChunk.chunk_type == "table",
                )
            )
            # 1. Section Scoping
            if target_sec:
                q = q.filter(DocumentChunk.item.ilike(f"%{target_sec.strip()}%"))

            # 2. Universal GAAP Statement Signature Matching
            if statement_type:
                if signatures:
                    q = q.filter(or_(*sig_clauses))
                else:
                    q = q.filter(DocumentChunk.content.ilike(f"%{statement_type}%"))

                # Exclude Index / TOC tables
                for pattern in INDEX_TABLE_PATTERNS:
                    q = q.filter(~DocumentChunk.content.ilike(pattern))

            # 3. Dynamic Keyword Filter
            if keyword_filter:
                kw = keyword_filter.strip()
                q = q.filter(
                    or_(
                        DocumentChunk.content.ilike(f"%{kw}%"),
                        DocumentChunk.sub_section.ilike(f"%{kw}%"),
                    )
                )

            # 4. Reading Order: Primary summary statements appear first
            return q.order_by(DocumentChunk.chunk_index.asc()).limit(limit).all()

        results = _execute_query(section_item)
        if not results and section_item:
            logger.info(
                f"retrieve_10k_tables: 0 tables found in '{section_item}', "
                f"falling back across all items for {ticker} FY{fiscal_year}"
            )
            results = _execute_query(None)

        output: List[TableChunkResult] = []
        for chunk, doc in results:
            meta = chunk.metadata_ or {}
            output.append(
                TableChunkResult(
                    chunk_id=str(chunk.id),
                    document_id=str(doc.id),
                    ticker=doc.ticker,
                    fiscal_year=doc.fiscal_year,
                    item=chunk.item or "Item 8",
                    breadcrumb=chunk.breadcrumb,
                    sub_section=chunk.sub_section or meta.get("table_title"),
                    table_markdown=chunk.content,
                    statement_type=normalized_statement,
                    row_count=meta.get("row_count", 0),
                    column_count=meta.get("column_count", 0),
                )
            )

        logger.info(
            f"retrieve_10k_tables: found {len(output)} tables for {ticker} FY{fiscal_year} "
            f"(statement={statement_type}, item={section_item}, keyword={keyword_filter})"
        )
        return output

    finally:
        if owns_db:
            db.close()


def retrieve_multiyear_financial_series(
    ticker: str,
    statement_type: str = "income_statement",
    section_item: Optional[str] = "Item 8",
    limit: int = 5,
    db: Optional[Session] = None,
) -> List[TableChunkResult]:
    """
    Queries across ALL available 10-K filings in the database for a given ticker,
    ordered by fiscal_year DESC.
    
    Retrieves the primary statement for each distinct fiscal year available,
    implementing the 'Latest Filing Precedence Rule' enabling quantitative agents
    to inspect multi-year trends and audit retrospective restatements across consecutive filings.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        statement_type: Target statement type ('income_statement', 'balance_sheet', 'cash_flow').
        section_item: Target 10-K section (defaults to 'Item 8').
        limit: Maximum number of yearly statements to return.
        db: Optional SQLAlchemy session.

    Returns:
        List of TableChunkResult sorted chronologically backwards (newest filing first).
    """
    owns_db = False
    if db is None:
        db = SessionLocal()
        owns_db = True

    try:
        # Find all distinct fiscal years for this ticker in descending order
        year_records = (
            db.query(Document.fiscal_year)
            .filter(Document.ticker == ticker.strip().upper())
            .distinct()
            .order_by(Document.fiscal_year.desc())
            .all()
        )
        available_years = [y[0] for y in year_records]

        output: List[TableChunkResult] = []
        for yr in available_years:
            yearly_tables = retrieve_10k_tables(
                ticker=ticker,
                fiscal_year=yr,
                statement_type=statement_type,
                section_item=section_item,
                limit=1,  # Pull the primary statement for this year
                db=db,
            )
            output.extend(yearly_tables)
            if len(output) >= limit:
                break

        logger.info(
            f"retrieve_multiyear_financial_series: retrieved {len(output)} yearly statements "
            f"for {ticker} across years {available_years}"
        )
        return output[:limit]

    finally:
        if owns_db:
            db.close()


# ==============================================================================
# 4. LangChain Agent Tool Decorator
# ==============================================================================
@tool
def retrieve_10k_tables_tool(
    ticker: str,
    fiscal_year: int,
    statement_type: Optional[str] = None,
    keyword_filter: Optional[str] = None,
    section_item: Optional[str] = "Item 8",
    limit: int = 1,
) -> List[Dict[str, Any]]:
    """
    Retrieve audited multi-year Markdown financial tables from a company's SEC 10-K filing.
    
    Use this tool to extract pristine financial statements (Income Statement, Balance Sheet,
    Cash Flows, or Segment data) without arithmetic hallucinations.

    Args:
        ticker: Stock symbol (e.g. 'AAPL', 'TSLA').
        fiscal_year: 4-digit fiscal year of the 10-K filing (e.g. 2025).
        statement_type: Optional statement name: 'income_statement', 'balance_sheet', 
                        'cash_flow', or 'segments'.
        keyword_filter: Optional text filter for specific line items or footnote disclosures.
        section_item: Filing item (default 'Item 8' for audited statements, or 'Item 7' for MD&A).
        limit: Maximum number of tables to return (default 1 to prioritize primary statements).

    Returns:
        List of structured table dictionaries containing clean markdown grids and citations.
    """
    results = retrieve_10k_tables(
        ticker=ticker,
        fiscal_year=fiscal_year,
        statement_type=statement_type,
        keyword_filter=keyword_filter,
        section_item=section_item,
        limit=limit,
    )
    return [r.model_dump() for r in results]


@tool
def retrieve_multiyear_financial_series_tool(
    ticker: str,
    statement_type: str = "income_statement",
    section_item: Optional[str] = "Item 8",
    limit: int = 2,
) -> List[Dict[str, Any]]:

    """
    Queries across ALL available 10-K filings in the database for a given ticker,
    ordered by fiscal_year DESC.

    Use this tool to compare financial statements across multiple consecutive 10-K filings,
    audit retrospective restatements, and verify multi-year historical trends
    under the Latest Filing Precedence Rule.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA').
        statement_type: Target statement type ('income_statement', 'balance_sheet', 'cash_flow').
        section_item: Target 10-K section (defaults to 'Item 8').
        limit: Maximum number of yearly statements to return.

    Returns:
        List of structured table dictionaries containing clean markdown grids and citations across multiple filings.
    """
    results = retrieve_multiyear_financial_series(
        ticker=ticker,
        statement_type=statement_type,
        section_item=section_item,
        limit=limit,
    )
    return [r.model_dump() for r in results]

