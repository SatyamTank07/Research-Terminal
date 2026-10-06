"""Universal 10-K Narrative Retrieval Tool (Hybrid RRF).

Enables qualitative agents (Business Strategist, Risk Analyst) and the Forecasting Analyst
to retrieve high-relevance narrative disclosures from SEC 10-K filings using Hybrid Search:
combining dense semantic vector embeddings (OpenAI text-embedding-3-small via pgvector HNSW)
with sparse PostgreSQL full-text search (TSVector via GIN) using Reciprocal Rank Fusion (RRF).

Includes Universal Section Fallback ensuring zero missed disclosures if a section tag differs,
while strictly locking ticker and fiscal year boundaries. Employs transaction SAVEPOINTs
(db.begin_nested()) to prevent transaction poisoning while preserving uncommitted caller state
in shared database sessions.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, text
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.database import SessionLocal
from app.models import Document, DocumentChunk

logger = logging.getLogger("finance_agent.tools.rag_narrative")

VALID_SEARCH_MODES = {"hybrid", "vector", "keyword"}

# Module-level cached embedder singleton
_CACHED_EMBEDDER: Optional[OpenAIEmbeddings] = None


# ==============================================================================
# 1. Structured Output Data Schema
# ==============================================================================
class NarrativeChunkResult(BaseModel):
    """Structured container for a retrieved 10-K narrative text chunk with source citations."""
    chunk_id: str = Field(..., description="Unique database UUID of the chunk for citations")
    document_id: str = Field(..., description="Parent 10-K document ID")
    ticker: str = Field(..., description="Stock symbol (e.g. AAPL, TSLA)")
    fiscal_year: int = Field(..., description="Fiscal year of the filing")
    item: str = Field(..., description="SEC 10-K Item (e.g. Item 1, Item 1A, Item 7)")
    breadcrumb: str = Field(..., description="Filing location trail")
    sub_section: Optional[str] = Field(None, description="Section or disclosure heading")
    content: str = Field(..., description="Clean markdown/text content of the narrative section")
    chunk_index: int = Field(..., description="Sequential position in the original filing")
    relevance_score: float = Field(0.0, description="RRF or similarity relevance score")


# ==============================================================================
# 2. Query Preprocessing & Helpers
# ==============================================================================
def _build_tsquery_expression(query_str: str):
    """
    Builds an OR-weighted PostgreSQL to_tsquery expression from a query string.
    Extracts alphanumeric words of length > 2 to avoid stopword / short-word noise,
    combining them with '|' (OR) so any matched key terminology boosts sparse ranking.
    """
    words = [re.sub(r"[^a-zA-Z0-9]", "", w) for w in query_str.split()]
    clean_words = [w for w in words if len(w) > 2]
    if not clean_words:
        return None
    ts_expr = " | ".join(clean_words)
    return func.to_tsquery("english", ts_expr)


def _get_embedder() -> Optional[OpenAIEmbeddings]:
    """Retrieves or instantiates a singleton OpenAIEmbeddings client."""
    global _CACHED_EMBEDDER
    if _CACHED_EMBEDDER is not None:
        return _CACHED_EMBEDDER

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set. Dense vector search disabled; falling back to TSVector.")
        return None
    try:
        _CACHED_EMBEDDER = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
        return _CACHED_EMBEDDER
    except Exception as e:
        logger.error(f"Error initializing OpenAIEmbeddings singleton: {e}")
        return None


# ==============================================================================
# 3. Core Retrieval Function (Hybrid RRF / Vector / Keyword)
# ==============================================================================
def retrieve_10k_narrative(
    ticker: str,
    fiscal_year: int,
    query: str,
    section_item: Optional[str] = None,
    limit: int = 5,
    search_mode: str = "hybrid",
    db: Optional[Session] = None,
) -> List[NarrativeChunkResult]:
    """
    Retrieves high-relevance 10-K narrative chunks for a target company and fiscal year.

    Implements Hybrid Reciprocal Rank Fusion (RRF) by default:
    - Dense Search: pgvector HNSW cosine distance using OpenAI text-embedding-3-small.
    - Sparse Search: PostgreSQL content_tsv full-text search with ts_rank_cd.
    - Universal Section Fallback: If section_item returns 0 results, searches across all items
      in that specific filing (ticker and fiscal_year remain strictly locked).
    - Savepoint Safety: Wraps internal query executions in SAVEPOINTS (db.begin_nested()).
      If an internal query fails, only the savepoint is rolled back, preventing connection poisoning
      while fully preserving any uncommitted staged work in shared caller sessions.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA', 'NVDA').
        fiscal_year: 4-digit fiscal year of the 10-K filing (e.g. 2025).
        query: Natural language search string or key topics.
        section_item: Optional SEC item filter (e.g. 'Item 1', 'Item 1A', 'Item 7').
        limit: Maximum number of narrative chunks to return (default 5, must be > 0).
        search_mode: 'hybrid' (default), 'vector', or 'keyword'.
        db: Optional existing SQLAlchemy session; if None, creates and closes its own.

    Returns:
        List of NarrativeChunkResult models with clean markdown content and source citations.
    """
    # 1. Validation Guards (reject booleans like True/False, non-ints, and non-positive numbers)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit must be a positive integer greater than 0, got {limit}")

    normalized_mode = search_mode.strip().lower() if search_mode else ""
    if normalized_mode not in VALID_SEARCH_MODES:
        raise ValueError(
            f"Invalid search_mode '{search_mode}'. Expected one of: {sorted(list(VALID_SEARCH_MODES))}."
        )

    owns_db = False
    if db is None:
        db = SessionLocal()
        owns_db = True

    try:
        normalized_ticker = ticker.strip().upper()
        clean_query = query.strip()

        # Step 1: Upfront Document Existence Verification
        # Fast exit before calling OpenAI embeddings if document doesn't exist
        doc_exists = (
            db.query(Document.id)
            .filter(
                Document.ticker == normalized_ticker,
                Document.fiscal_year == fiscal_year,
            )
            .first()
        )
        if not doc_exists:
            logger.warning(
                f"retrieve_10k_narrative: No 10-K document found in database for ticker '{normalized_ticker}' "
                f"and fiscal year {fiscal_year}."
            )
            return []

        embedder = _get_embedder()

        # Generate query vector if vector or hybrid search is requested
        query_vector: Optional[List[float]] = None
        if normalized_mode in ("hybrid", "vector") and embedder is not None:
            try:
                query_vector = embedder.embed_query(clean_query)
            except Exception as e:
                logger.warning(f"Failed to generate query embedding: {e}. Falling back to sparse search.")

        # Candidate limit for RRF pooling
        candidate_pool_size = max(limit * 4, 25)

        def _fetch_candidates(target_sec: Optional[str]):
            """Queries dense and sparse candidates using eager loading and savepoint-scoped error recovery."""
            dense_candidates: List[DocumentChunk] = []
            sparse_candidates: List[DocumentChunk] = []

            # Base filter for this filing's narrative chunks
            base_filters = [
                Document.ticker == normalized_ticker,
                Document.fiscal_year == fiscal_year,
                DocumentChunk.chunk_type == "narrative",
            ]
            if target_sec:
                base_filters.append(DocumentChunk.item.ilike(f"%{target_sec.strip()}%"))

            # 1. Dense Vector Query (scoped inside SAVEPOINT to protect shared transactions)
            if query_vector is not None and normalized_mode in ("hybrid", "vector"):
                savepoint = db.begin_nested()
                try:
                    dense_q = (
                        db.query(DocumentChunk)
                        .options(joinedload(DocumentChunk.document))
                        .join(Document, Document.id == DocumentChunk.document_id)
                        .filter(*base_filters)
                        .filter(DocumentChunk.embedding.isnot(None))
                        .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
                        .limit(candidate_pool_size)
                    )
                    dense_candidates = dense_q.all()
                    savepoint.commit()
                except Exception as e:
                    savepoint.rollback()  # Resets savepoint without rolling back caller's outer transaction
                    logger.error(f"Error executing pgvector dense query: {e}. Rolled back savepoint.")
                    dense_candidates = []

            # 2. Sparse TSVector Query (scoped inside SAVEPOINT to protect shared transactions)
            if normalized_mode in ("hybrid", "keyword") or query_vector is None:
                tsq_expr = _build_tsquery_expression(clean_query)
                if tsq_expr is not None:
                    savepoint = db.begin_nested()
                    try:
                        sparse_q = (
                            db.query(DocumentChunk)
                            .options(joinedload(DocumentChunk.document))
                            .join(Document, Document.id == DocumentChunk.document_id)
                            .filter(*base_filters)
                            .filter(DocumentChunk.content_tsv.op("@@")(tsq_expr))
                            .order_by(func.ts_rank_cd(DocumentChunk.content_tsv, tsq_expr).desc())
                            .limit(candidate_pool_size)
                        )
                        sparse_candidates = sparse_q.all()
                        savepoint.commit()
                    except Exception as e:
                        savepoint.rollback()  # Resets savepoint without rolling back caller's outer transaction
                        logger.warning(f"Error executing PostgreSQL TSVector search: {e}. Rolled back savepoint.")
                        sparse_candidates = []

            return dense_candidates, sparse_candidates

        # Step 2: Initial Attempt with requested section filter
        dense_list, sparse_list = _fetch_candidates(section_item)

        # Step 3: Universal Section Fallback (if section_item was given but returned 0 results)
        if not dense_list and not sparse_list and section_item:
            logger.info(
                f"retrieve_10k_narrative: 0 chunks found in '{section_item}' for {normalized_ticker} FY{fiscal_year}. "
                f"Applying Universal Section Fallback across all sections of this filing."
            )
            dense_list, sparse_list = _fetch_candidates(None)

        # Step 4: Rank Fusion or Mode Selection
        candidate_map: Dict[str, DocumentChunk] = {}
        rrf_scores: Dict[str, float] = {}
        k_rrf = 60.0

        # Dense ranking scores
        for rank_idx, chunk in enumerate(dense_list):
            cid = str(chunk.id)
            candidate_map[cid] = chunk
            if normalized_mode == "vector":
                rrf_scores[cid] = 1.0 / (rank_idx + 1)
            else:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank_idx + 1))

        # Sparse ranking scores
        for rank_idx, chunk in enumerate(sparse_list):
            cid = str(chunk.id)
            candidate_map[cid] = chunk
            if normalized_mode == "keyword":
                rrf_scores[cid] = 1.0 / (rank_idx + 1)
            else:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank_idx + 1))

        if not candidate_map:
            logger.warning(
                f"retrieve_10k_narrative: No narrative chunks found for {normalized_ticker} FY{fiscal_year} "
                f"(query='{clean_query}', section='{section_item}', mode='{normalized_mode}')"
            )
            return []

        # Sort candidates by combined score descending
        sorted_chunk_ids = sorted(candidate_map.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_chunk_ids = sorted_chunk_ids[:limit]

        results: List[NarrativeChunkResult] = []
        for cid in top_chunk_ids:
            chunk = candidate_map[cid]
            doc = chunk.document
            score = round(rrf_scores[cid], 5)

            results.append(
                NarrativeChunkResult(
                    chunk_id=str(chunk.id),
                    document_id=str(doc.id if doc else chunk.document_id),
                    ticker=doc.ticker if doc else normalized_ticker,
                    fiscal_year=doc.fiscal_year if doc else fiscal_year,
                    item=chunk.item or "Narrative",
                    breadcrumb=chunk.breadcrumb,
                    sub_section=chunk.sub_section,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    relevance_score=score,
                )
            )

        logger.info(
            f"retrieve_10k_narrative: retrieved {len(results)} chunks for {normalized_ticker} FY{fiscal_year} "
            f"(mode={normalized_mode}, item={section_item}, top_score={results[0].relevance_score if results else 0.0})"
        )
        return results

    except Exception as e:
        if owns_db:
            db.rollback()
        logger.error(f"retrieve_10k_narrative unhandled error for {ticker} FY{fiscal_year}: {e}")
        raise e
    finally:
        if owns_db:
            db.close()


# ==============================================================================
# 4. LangChain Agent Tool Decorator
# ==============================================================================
@tool
def retrieve_10k_narrative_tool(
    ticker: str,
    fiscal_year: int,
    query: str,
    section_item: Optional[str] = None,
    limit: int = 5,
    search_mode: str = "hybrid",
) -> List[Dict[str, Any]]:
    """
    Retrieve audited narrative text disclosures from a company's SEC 10-K filing.

    Use this tool to extract high-relevance qualitative context on business model,
    economic moat, product segments (Item 1), risk factors (Item 1A), or management's
    outlook and forward expectations (Item 7 MD&A).

    Employs Hybrid Reciprocal Rank Fusion (dense vector + full-text search) to ensure
    both conceptual understanding and exact keyword/entity precision.

    Args:
        ticker: Stock symbol (e.g. 'AAPL', 'TSLA', 'NVDA').
        fiscal_year: 4-digit fiscal year of the 10-K filing (e.g. 2025).
        query: Natural language topic or concepts to search for.
        section_item: Optional 10-K section filter: 'Item 1' (Business),
                      'Item 1A' (Risk Factors), or 'Item 7' (MD&A).
        limit: Maximum number of narrative chunks to return (default 5, must be > 0).
        search_mode: Search strategy: 'hybrid' (default), 'vector', or 'keyword'.

    Returns:
        List of structured narrative chunk dictionaries with clean markdown content and citations.
    """
    results = retrieve_10k_narrative(
        ticker=ticker,
        fiscal_year=fiscal_year,
        query=query,
        section_item=section_item,
        limit=limit,
        search_mode=search_mode,
    )
    return [r.model_dump() for r in results]
