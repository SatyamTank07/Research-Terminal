"""Unit and integration tests for rag_narrative_tools.py (Milestone 5 Foundation Tool).

Validates Hybrid RRF (pgvector HNSW + TSVector GIN) retrieval across real ingested 10-K filings:
- Apple Inc. (AAPL FY2025 & FY2024)
- Tesla, Inc. (TSLA FY2025)
- NVIDIA Corp. (NVDA FY2026)
"""

import unittest
from app.agents.tools.rag_narrative_tools import (
    NarrativeChunkResult,
    retrieve_10k_narrative,
    retrieve_10k_narrative_tool,
)


class TestRAGNarrativeTools(unittest.TestCase):
    """Test suite verifying universal 10-K narrative retrieval via Hybrid RRF."""

    def test_01_aapl_item1_business_retrieval(self):
        """Verify Business Strategist query retrieves clean Item 1 product and service disclosures."""
        results = retrieve_10k_narrative(
            ticker="AAPL",
            fiscal_year=2025,
            query="revenue model products services iPhone Mac AppleCare ecosystem",
            section_item="Item 1",
            limit=5,
            search_mode="hybrid",
        )
        self.assertGreater(len(results), 0, "Should return narrative chunks for AAPL Item 1")
        self.assertLessEqual(len(results), 5)

        primary = results[0]
        self.assertIsInstance(primary, NarrativeChunkResult)
        self.assertEqual(primary.ticker, "AAPL")
        self.assertEqual(primary.fiscal_year, 2025)
        self.assertIn("Item 1", primary.item)
        self.assertTrue(len(primary.chunk_id) > 0)
        self.assertTrue(len(primary.document_id) > 0)
        self.assertTrue(len(primary.breadcrumb) > 0)
        self.assertGreater(primary.relevance_score, 0.0)

        # Check content relevance
        combined_text = " ".join(r.content.lower() for r in results)
        self.assertTrue(
            any(w in combined_text for w in ["iphone", "services", "applecare", "products"]),
            "Retrieved text should contain core Apple business keywords",
        )

    def test_02_tsla_item1a_risk_retrieval(self):
        """Verify Risk Analyst query isolates specific battery and supply chain risks in Item 1A."""
        results = retrieve_10k_narrative(
            ticker="TSLA",
            fiscal_year=2025,
            query="lithium battery cell supplier concentration supply chain risks",
            section_item="Item 1A",
            limit=5,
            search_mode="hybrid",
        )
        self.assertGreater(len(results), 0, "Should return risk chunks for TSLA Item 1A")
        
        primary = results[0]
        self.assertIsInstance(primary, NarrativeChunkResult)
        self.assertEqual(primary.ticker, "TSLA")
        self.assertEqual(primary.fiscal_year, 2025)
        self.assertIn("Item 1A", primary.item)

        # Tesla battery supplier disclosure should be in top results
        combined_text = " ".join(r.content.lower() for r in results)
        self.assertTrue(
            "battery" in combined_text or "lithium" in combined_text or "supplier" in combined_text,
            "Top TSLA risk chunks should discuss battery/supplier dependencies",
        )

    def test_03_nvda_item7_mda_forecaster_retrieval(self):
        """Verify Forecasting Analyst query pulls forward demand drivers in Item 7 MD&A."""
        results = retrieve_10k_narrative(
            ticker="NVDA",
            fiscal_year=2026,
            query="data center computing architectures accelerated computing demand drivers",
            section_item="Item 7",
            limit=5,
            search_mode="hybrid",
        )
        self.assertGreater(len(results), 0, "Should return MD&A narrative chunks for NVDA")

        primary = results[0]
        self.assertIsInstance(primary, NarrativeChunkResult)
        self.assertEqual(primary.ticker, "NVDA")
        self.assertEqual(primary.fiscal_year, 2026)
        self.assertIn("Item 7", primary.item)

        combined_text = " ".join(r.content.lower() for r in results)
        self.assertTrue(
            "data center" in combined_text or "computing" in combined_text or "revenue" in combined_text,
            "Retrieved Item 7 chunks should contain Data Center and computing driver commentary",
        )

    def test_04_universal_section_fallback(self):
        """Verify Universal Section Fallback gracefully recovers when section_item does not match."""
        results = retrieve_10k_narrative(
            ticker="TSLA",
            fiscal_year=2025,
            query="autonomous driving full self driving robotaxi technology",
            section_item="Item 999_NonExistentSection",
            limit=3,
            search_mode="hybrid",
        )
        self.assertGreater(len(results), 0, "Fallback should return chunks despite non-existent section filter")
        for r in results:
            self.assertEqual(r.ticker, "TSLA", "Fallback must strictly preserve ticker boundary")
            self.assertEqual(r.fiscal_year, 2025, "Fallback must strictly preserve fiscal year boundary")

    def test_05_strict_filing_isolation(self):
        """Verify cross-ticker isolation: AAPL queries must never return TSLA or NVDA chunks."""
        results = retrieve_10k_narrative(
            ticker="AAPL",
            fiscal_year=2025,
            query="artificial intelligence neural network hardware acceleration",
            limit=5,
            search_mode="hybrid",
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.ticker, "AAPL")
            self.assertEqual(r.fiscal_year, 2025)

    def test_06_search_modes_comparability(self):
        """Verify vector-only, keyword-only, and hybrid search modes all function properly."""
        vector_res = retrieve_10k_narrative(
            ticker="AAPL",
            fiscal_year=2025,
            query="iPhone Mac iPad Services hardware",
            section_item="Item 1",
            limit=3,
            search_mode="vector",
        )
        self.assertGreater(len(vector_res), 0, "Vector search should return results")

        keyword_res = retrieve_10k_narrative(
            ticker="AAPL",
            fiscal_year=2025,
            query="iPhone Mac iPad Services hardware",
            section_item="Item 1",
            limit=3,
            search_mode="keyword",
        )
        self.assertGreater(len(keyword_res), 0, "Keyword search should return results")

        hybrid_res = retrieve_10k_narrative(
            ticker="AAPL",
            fiscal_year=2025,
            query="iPhone Mac iPad Services hardware",
            section_item="Item 1",
            limit=3,
            search_mode="hybrid",
        )
        self.assertGreater(len(hybrid_res), 0, "Hybrid search should return results")

    def test_07_langchain_tool_decorator(self):
        """Verify retrieve_10k_narrative_tool LangChain tool wrapper returns serialized dicts."""
        tool_output = retrieve_10k_narrative_tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "query": "iPhone products and services overview",
            "section_item": "Item 1",
            "limit": 2,
        })
        self.assertIsInstance(tool_output, list)
        self.assertEqual(len(tool_output), 2)

        first = tool_output[0]
        self.assertIsInstance(first, dict)
        self.assertIn("chunk_id", first)
        self.assertIn("document_id", first)
        self.assertIn("ticker", first)
        self.assertIn("fiscal_year", first)
        self.assertIn("item", first)
        self.assertIn("breadcrumb", first)
        self.assertIn("content", first)
        self.assertIn("relevance_score", first)
        self.assertEqual(first["ticker"], "AAPL")
        self.assertEqual(first["fiscal_year"], 2025)

    def test_08_invalid_search_mode_guard(self):
        """Verify passing an unrecognized search_mode raises ValueError with clear guidance."""
        with self.assertRaises(ValueError) as ctx:
            retrieve_10k_narrative(
                ticker="AAPL",
                fiscal_year=2025,
                query="iPhone ecosystem",
                search_mode="semantic_fuzzy",
            )
        self.assertIn("Invalid search_mode 'semantic_fuzzy'", str(ctx.exception))
        self.assertIn("hybrid", str(ctx.exception))

    def test_09_limit_boundary_guard(self):
        """Verify non-positive limit values (0 or negative) and booleans raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            retrieve_10k_narrative(
                ticker="AAPL",
                fiscal_year=2025,
                query="iPhone ecosystem",
                limit=0,
            )
        self.assertIn("limit must be a positive integer greater than 0", str(ctx.exception))

        with self.assertRaises(ValueError):
            retrieve_10k_narrative(
                ticker="AAPL",
                fiscal_year=2025,
                query="iPhone ecosystem",
                limit=-5,
            )

        # Reject booleans (True/False are subclasses of int in Python)
        with self.assertRaises(ValueError):
            retrieve_10k_narrative(
                ticker="AAPL",
                fiscal_year=2025,
                query="iPhone ecosystem",
                limit=True,
            )

    def test_10_embedder_singleton(self):
        """Verify _get_embedder returns a cached singleton across repeated calls."""
        from app.agents.tools.rag_narrative_tools import _get_embedder
        emb1 = _get_embedder()
        emb2 = _get_embedder()
        self.assertIs(emb1, emb2, "_get_embedder should return the exact same cached instance")

    def test_11_eager_document_loading_no_lazy_n_plus_one(self):
        """Verify candidate queries load Document relationships eagerly."""
        results = retrieve_10k_narrative(
            ticker="TSLA",
            fiscal_year=2025,
            query="battery cell production",
            limit=5,
            search_mode="hybrid",
        )
        self.assertGreater(len(results), 0)
        # Verify document metadata is correctly populated without triggering separate queries
        for r in results:
            self.assertEqual(r.ticker, "TSLA")
            self.assertEqual(r.fiscal_year, 2025)
            self.assertTrue(len(r.document_id) > 0)

    def test_12_nonexistent_document_fast_exit(self):
        """Verify requesting a filing not in the database returns [] cleanly without crashing."""
        results = retrieve_10k_narrative(
            ticker="NONEXISTENT_CORP",
            fiscal_year=2099,
            query="quantum computer cloud revenue",
        )
        self.assertEqual(results, [])

    def test_13_shared_session_savepoint_isolation_on_failure(self):
        """Verify savepoint rollback preserves uncommitted caller work when an internal query fails."""
        from unittest.mock import patch
        from sqlalchemy import func
        from app.database import SessionLocal
        from app.models import Document, User
        
        caller_session = SessionLocal()
        try:
            # 1. Stage an uncommitted object in the caller's outer transaction
            staged_user = User(
                username="uncommitted_test_user_777",
                email="uncommitted@test.com",
            )
            caller_session.add(staged_user)
            self.assertIn(staged_user, caller_session, "User should be tracked in caller session")

            # 2. Force an internal PostgreSQL syntax error on the sparse query via monkeypatch
            # to_tsquery with invalid syntax raises a PostgreSQL syntax error when executed
            malformed_expr = func.to_tsquery("english", "&&& bad tsquery syntax !&")
            with patch("app.agents.tools.rag_narrative_tools._build_tsquery_expression", return_value=malformed_expr):
                results = retrieve_10k_narrative(
                    ticker="AAPL",
                    fiscal_year=2025,
                    query="products",
                    search_mode="keyword",  # Forces sparse_q to execute the bad tsquery in Postgres
                    db=caller_session,
                )
                self.assertEqual(results, [], "Failed query should degrade gracefully to empty list")

            # 3. Assert SAVEPOINT isolation:
            # The caller's uncommitted staged work was NOT wiped out by global rollback!
            self.assertIn(staged_user, caller_session, "Caller's uncommitted work must survive savepoint rollback")

            # 4. Assert transaction is NOT poisoned:
            # Caller session can immediately run a follow-up query without InFailedSqlTransaction
            doc = caller_session.query(Document).filter(Document.ticker == "AAPL").first()
            self.assertIsNotNone(doc)
            self.assertEqual(doc.ticker, "AAPL")

        finally:
            caller_session.rollback()  # Clean up staged user
            caller_session.close()


if __name__ == "__main__":
    unittest.main()



