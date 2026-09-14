import os
import time
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, delete

from app.models import Document, DocumentChunk
from app.services.ingestion.sec_parser import SECParser
from app.services.ingestion.chunker import SECChunker
from app.services.ingestion.embedder import SECBatchEmbedder

logger = logging.getLogger("finance_agent.ingestion.pipeline")


class SECIngestionPipeline:
    """End-to-end ingestion pipeline for SEC filings into PostgreSQL."""

    def __init__(self, db: Session):
        self.db = db
        self.parser = SECParser()
        self.chunker = SECChunker()
        self.embedder = SECBatchEmbedder()

    def run(self, file_path: str, progress_callback=None) -> Dict[str, Any]:
        """Executes the ingestion pipeline on the specified HTML filing file."""
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Filing file not found: {file_path}")

        logger.info(f"Starting ingestion pipeline for: {file_path}")

        if progress_callback:
            progress_callback({
                "stage": "parsing",
                "progress": 15,
                "message": f"Reading & parsing iXBRL structure ({os.path.basename(file_path)})...",
            })

        # Step 1: Parse iXBRL
        t0 = time.time()
        metadata, sections = self.parser.parse(file_path)
        parse_duration = time.time() - t0
        logger.info(f"Parsed filing in {parse_duration:.2f}s: {metadata.company_name} ({metadata.ticker}) FY{metadata.fiscal_year}")

        if progress_callback:
            progress_callback({
                "stage": "chunking",
                "progress": 35,
                "message": f"Identified {metadata.company_name} ({metadata.ticker}) FY{metadata.fiscal_year}. Chunking {len(sections)} sections...",
            })

        # Step 2: Metadata-based Chunking
        t0 = time.time()
        chunks = self.chunker.chunk_document(metadata, sections)
        chunk_duration = time.time() - t0
        logger.info(f"Generated {len(chunks)} chunks in {chunk_duration:.2f}s")

        table_count = sum(1 for c in chunks if c.chunk_type == "table")
        narrative_count = sum(1 for c in chunks if c.chunk_type == "narrative")

        if progress_callback:
            progress_callback({
                "stage": "embedding",
                "progress": 50,
                "message": f"Generated {len(chunks)} chunks ({table_count} tables). Generating pgvector embeddings...",
            })

        # Step 3: Batch Embeddings (for Search Faces)
        t0 = time.time()
        search_faces = [c.embedding_input for c in chunks]

        def on_batch(b_num, total_b):
            if progress_callback:
                pct = 50 + int((b_num / total_b) * 35)
                progress_callback({
                    "stage": "embedding",
                    "progress": pct,
                    "message": f"Computing pgvector embeddings (Batch {b_num}/{total_b})...",
                })

        embeddings = self.embedder.embed_texts(search_faces, on_batch_progress=on_batch)
        embed_duration = time.time() - t0
        logger.info(f"Generated {len(embeddings)} embeddings in {embed_duration:.2f}s")

        if progress_callback:
            progress_callback({
                "stage": "storing",
                "progress": 90,
                "message": "Persisting chunks and HNSW/GIN indexes in PostgreSQL...",
            })

        # Step 4: PostgreSQL Storage
        t0 = time.time()

        # Idempotency: Remove previous ingestion of the exact same filing if it exists
        existing_doc = (
            self.db.query(Document)
            .filter(
                Document.ticker == metadata.ticker,
                Document.filing_type == metadata.filing_type,
                Document.fiscal_year == metadata.fiscal_year,
            )
            .first()
        )
        if existing_doc:
            logger.info(f"Replacing existing document record for {metadata.ticker} FY{metadata.fiscal_year} (ID: {existing_doc.id})")
            self.db.delete(existing_doc)
            self.db.flush()

        # Create Document record
        doc = Document(
            ticker=metadata.ticker,
            company_name=metadata.company_name,
            filing_type=metadata.filing_type,
            fiscal_year=metadata.fiscal_year,
            period_end_date=metadata.period_end_date,
            source_filename=metadata.source_filename,
            total_chunks=len(chunks),
        )
        self.db.add(doc)
        self.db.flush()  # Populates doc.id

        # Insert DocumentChunk records
        chunk_models = []
        for i, chunk_item in enumerate(chunks):
            vector_val = embeddings[i] if embeddings and i < len(embeddings) else None

            chunk_obj = DocumentChunk(
                document_id=doc.id,
                part=chunk_item.part,
                item=chunk_item.item,
                sub_section=chunk_item.sub_section,
                breadcrumb=chunk_item.breadcrumb,
                chunk_type=chunk_item.chunk_type,
                chunk_index=chunk_item.chunk_index,
                content=chunk_item.content,
                metadata_=chunk_item.metadata,
                content_tsv=func.to_tsvector("english", chunk_item.breadcrumb + " " + chunk_item.content),
                embedding=vector_val,
            )
            chunk_models.append(chunk_obj)

        self.db.add_all(chunk_models)
        self.db.commit()
        db_duration = time.time() - t0

        total_duration = time.time() - start_time
        logger.info(f"Ingestion completed in {total_duration:.2f}s! Saved {len(chunk_models)} chunks in PostgreSQL.")

        result = {
            "status": "success",
            "document_id": doc.id,
            "ticker": doc.ticker,
            "company_name": doc.company_name,
            "filing_type": doc.filing_type,
            "fiscal_year": doc.fiscal_year,
            "period_end_date": doc.period_end_date,
            "total_chunks": len(chunks),
            "narrative_chunks": narrative_count,
            "table_chunks": table_count,
            "timing": {
                "parse_seconds": round(parse_duration, 2),
                "chunk_seconds": round(chunk_duration, 2),
                "embed_seconds": round(embed_duration, 2),
                "storage_seconds": round(db_duration, 2),
                "total_seconds": round(total_duration, 2),
            },
        }

        if progress_callback:
            progress_callback({
                "stage": "completed",
                "progress": 100,
                "message": f"Successfully indexed {metadata.ticker} FY{metadata.fiscal_year} ({len(chunks)} chunks ready)!",
                "data": result,
            })

        return result

