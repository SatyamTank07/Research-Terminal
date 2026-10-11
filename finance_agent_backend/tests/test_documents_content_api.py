"""Filing viewer API: whole-filing content and re-ingestion locate fallback (real PostgreSQL)."""

import unittest

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import Document, DocumentChunk

TICKER = "ZZCITE"
CHUNKS = [  # inserted out of order on purpose
    (2, "PART II", "Item 7", "ZZ Co (ZZCITE) > 10-K FY2099 > PART II > Item 7: MD&A", "narrative"),
    (0, "PART I", "Item 1", "ZZ Co (ZZCITE) > 10-K FY2099 > PART I > Item 1: Business", "narrative"),
    (1, "PART I", "Item 1", "ZZ Co (ZZCITE) > 10-K FY2099 > PART I > Item 1: Business", "narrative"),
    (3, "PART II", "Item 8", "ZZ Co (ZZCITE) > 10-K FY2099 > PART II > Item 8: Financial Statements", "table"),
]


def _purge(db):
    for doc in db.query(Document).filter(Document.ticker == TICKER).all():
        db.delete(doc)
    db.commit()


class TestFilingContentApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        db = SessionLocal()
        try:
            _purge(db)
            doc = Document(ticker=TICKER, company_name="ZZ Co", filing_type="10-K", fiscal_year=2099,
                           source_filename="zzcite-20991231.htm", total_chunks=len(CHUNKS))
            db.add(doc)
            db.flush()
            cls.chunk_ids = {}
            for idx, part, item, crumb, kind in CHUNKS:
                chunk = DocumentChunk(document_id=doc.id, part=part, item=item, breadcrumb=crumb, chunk_type=kind,
                                      chunk_index=idx, content=f"Chunk {idx} body", metadata_={"probe": True})
                db.add(chunk)
                db.flush()
                cls.chunk_ids[idx] = chunk.id
            db.commit()
            cls.document_id = doc.id
        finally:
            db.close()

    @classmethod
    def tearDownClass(cls):
        db = SessionLocal()
        try:
            _purge(db)
        finally:
            db.close()

    def test_content_is_ordered_and_lean(self):
        res = self.client.get(f"/api/documents/{self.document_id}/content")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["document"]["ticker"], TICKER)
        self.assertEqual([c["chunk_index"] for c in body["chunks"]], [0, 1, 2, 3])
        self.assertEqual([c["id"] for c in body["chunks"]], [self.chunk_ids[i] for i in range(4)])
        for chunk in body["chunks"]:
            self.assertEqual(set(chunk), {"id", "chunk_index", "part", "item", "breadcrumb", "chunk_type", "content"})

    def test_toc_lists_first_chunk_of_each_item(self):
        toc = self.client.get(f"/api/documents/{self.document_id}/content").json()["toc"]
        self.assertEqual(toc, [
            {"part": "PART I", "item": "Item 1", "title": "Item 1: Business", "chunk_index": 0},
            {"part": "PART II", "item": "Item 7", "title": "Item 7: MD&A", "chunk_index": 2},
            {"part": "PART II", "item": "Item 8", "title": "Item 8: Financial Statements", "chunk_index": 3},
        ])

    def test_content_unknown_document_404(self):
        self.assertEqual(self.client.get("/api/documents/does-not-exist/content").status_code, 404)

    def test_locate_prefers_same_breadcrumb_nearest_index(self):
        res = self.client.get("/api/documents/locate", params={
            "ticker": TICKER.lower(), "fiscal_year": 2099, "breadcrumb": CHUNKS[1][3], "chunk_index": 1,
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"document_id": self.document_id, "chunk_id": self.chunk_ids[1], "exact": False})

    def test_locate_falls_back_to_item_then_index(self):
        by_item = self.client.get("/api/documents/locate", params={
            "ticker": TICKER, "fiscal_year": 2099, "breadcrumb": "gone", "item": "Item 8", "chunk_index": 0,
        }).json()
        self.assertEqual(by_item["chunk_id"], self.chunk_ids[3])
        by_index = self.client.get("/api/documents/locate", params={
            "ticker": TICKER, "fiscal_year": 2099, "chunk_index": 2,
        }).json()
        self.assertEqual(by_index["chunk_id"], self.chunk_ids[2])

    def test_locate_missing_filing_404(self):
        res = self.client.get("/api/documents/locate", params={"ticker": TICKER, "fiscal_year": 1900})
        self.assertEqual(res.status_code, 404)
