"""Tests for the forecast context: the Item 7 (MD&A) fetch and its empty fallback (retrieval mocked)."""

import unittest
from unittest.mock import patch

from app.agents.specialized.forecasting_analyst.context import MDNA_CHUNK_LIMIT, MDNA_QUERY, fetch_mdna_chunks
from tests.forecast_fixtures import GUIDANCE_CHUNK, make_chunk

RETRIEVE = "app.agents.specialized.forecasting_analyst.context.retrieve_10k_narrative"


class TestFetchMdnaChunks(unittest.TestCase):
    def test_searches_item_7_with_the_outlook_query(self):
        chunks = [make_chunk("c1", GUIDANCE_CHUNK)]
        with patch(RETRIEVE, return_value=chunks) as retrieve:
            self.assertEqual(fetch_mdna_chunks("AAPL", 2025), chunks)
        retrieve.assert_called_once_with(
            ticker="AAPL", fiscal_year=2025, query=MDNA_QUERY, section_item="Item 7", limit=MDNA_CHUNK_LIMIT,
        )

    def test_failed_retrieval_returns_no_excerpts(self):
        with patch(RETRIEVE, side_effect=RuntimeError("db down")):
            self.assertEqual(fetch_mdna_chunks("AAPL", 2025), [])


if __name__ == "__main__":
    unittest.main()
