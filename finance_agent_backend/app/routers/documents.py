import os
import json
import queue
import threading
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import Document, DocumentChunk
from app.services.ingestion.pipeline import SECIngestionPipeline

logger = logging.getLogger("finance_agent.routers.documents")

router = APIRouter(prefix="/api/documents", tags=["Documents"])


class IngestDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Path or filename of the SEC HTML filing (e.g. 'aapl-20250927.htm')")


def resolve_filing_path(file_path: str) -> str:
    """Finds the filing file across possible local/container relative locations."""
    if os.path.exists(file_path):
        return os.path.abspath(file_path)

    # Candidate directories
    candidates = [
        os.path.join(os.getcwd(), file_path),
        os.path.join(os.getcwd(), "..", file_path),
        os.path.join("/workspace", file_path),
        os.path.join("/workspace", os.path.basename(file_path)),
        os.path.join("/app", file_path),
        os.path.join("/app", "..", file_path),
        os.path.join("c:/Satyam/Projects/finance_agent", file_path),
        os.path.join("c:\\Satyam\\Projects\\finance_agent", file_path),
    ]

    for cand in candidates:
        norm = os.path.normpath(cand)
        if os.path.exists(norm):
            return norm

    raise FileNotFoundError(f"Filing file could not be found at '{file_path}' or any known search paths.")


@router.post("/ingest", status_code=status.HTTP_200_OK)
def ingest_filing(request: IngestDocumentRequest, db: Session = Depends(get_db)):
    """Ingests an SEC Form 10-K / 10-Q filing into PostgreSQL with pgvector embeddings."""
    try:
        resolved_path = resolve_filing_path(request.file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    try:
        pipeline = SECIngestionPipeline(db=db)
        result = pipeline.run(resolved_path)
        return result
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline failed: {str(e)}",
        )


@router.post("/upload")
def upload_and_ingest_filing(file: UploadFile = File(...)):
    """Accepts an uploaded .htm SEC filing and streams real-time SSE progress events."""
    if not file.filename.lower().endswith(".htm"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .htm SEC filing files are supported.",
        )

    # Save uploaded file to cache folder
    upload_dir = "/tmp/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)

    with open(temp_path, "wb") as f:
        f.write(file.file.read())

    def event_stream():
        event_queue = queue.Queue()

        def background_worker():
            worker_db = SessionLocal()
            try:
                def progress_reporter(event_dict):
                    event_queue.put(event_dict)

                pipeline = SECIngestionPipeline(db=worker_db)
                res = pipeline.run(temp_path, progress_callback=progress_reporter)
                event_queue.put({"_worker_done": True})
            except Exception as ex:
                logger.exception(f"Background upload ingestion failed: {ex}")
                event_queue.put({"stage": "error", "progress": 0, "message": str(ex)})
                event_queue.put({"_worker_done": True})
            finally:
                worker_db.close()

        # Emit initial upload received event
        yield f"data: {json.dumps({'stage': 'uploaded', 'progress': 10, 'message': f'Filing uploaded: {file.filename}'})}\n\n"

        t = threading.Thread(target=background_worker)
        t.daemon = True
        t.start()

        while True:
            try:
                event = event_queue.get(timeout=60.0)
                if event.get("_worker_done"):
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Keep-alive heartbeat
                yield f": heartbeat\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Deletes an ingested filing and its associated vector chunks."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    db.delete(doc)
    db.commit()
    return {"status": "deleted", "document_id": document_id}


@router.get("/", status_code=status.HTTP_200_OK)
def list_documents(db: Session = Depends(get_db)):
    """Returns a list of all ingested SEC filing documents."""
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [d.to_dict() for d in docs]


@router.get("/{document_id}", status_code=status.HTTP_200_OK)
def get_document_details(document_id: str, db: Session = Depends(get_db)):
    """Returns details and chunk summary for a specific ingested document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Sample stats
    narrative_count = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id, DocumentChunk.chunk_type == "narrative")
        .count()
    )
    table_count = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id, DocumentChunk.chunk_type == "table")
        .count()
    )

    doc_data = doc.to_dict()
    doc_data["narrative_chunks"] = narrative_count
    doc_data["table_chunks"] = table_count
    return doc_data
