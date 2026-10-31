"""Quote and number checks that keep LLM-proposed evidence tied to the filing chunks it cites."""

import unittest

from app.agents.shared.quotes import find_quote_chunk, numbers_supported, verify_quote

CONCENTRATION = (
    "We receive a significant amount of our revenue from a limited number of customers. "
    "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
    "another direct customer represented 14% of total revenue, all of which were primarily attributable "
    "to the Compute & Networking segment."
)

NEAR = "sales to one customer represented 22% of total revenue and sales to another direct customer represented 14%"
EXACT = "sales to one direct customer represented 22% of total revenue"
UNRELATED = "Our offices are leased under agreements that expire at various dates through the next decade."


def _pool(*chunks):
    return {cid: {"chunk_id": cid, "content": text, "item": "Item 1A", "breadcrumb": None} for cid, text in chunks}


class TestQuoteVerification(unittest.TestCase):
    def test_exact_and_normalized_quotes_pass(self):
        quote = "sales to one direct customer represented 22% of total revenue"
        self.assertTrue(verify_quote(quote, CONCENTRATION))
        self.assertTrue(verify_quote(quote.upper().replace(" ", "   "), CONCENTRATION))

    def test_curly_quotes_and_dashes_are_normalized(self):
        text = "The Company’s products — including chips — face export licensing requirements in China."
        self.assertTrue(verify_quote("The Company's products - including chips - face export licensing requirements", text))

    def test_ellipsis_segments_each_must_match(self):
        quote = "sales to one direct customer represented 22% ... another direct customer represented 14% of total revenue"
        self.assertTrue(verify_quote(quote, CONCENTRATION))
        bad = "sales to one direct customer represented 22% ... another customer represented 40% of total revenue"
        self.assertFalse(verify_quote(bad, CONCENTRATION))

    def test_near_verbatim_passes_only_with_fuzzy(self):
        quote = "sales to one customer represented 22% of total revenue and sales to another direct customer represented 14%"
        self.assertTrue(verify_quote(quote, CONCENTRATION))
        self.assertFalse(verify_quote(quote, CONCENTRATION, allow_fuzzy=False))

    def test_fabricated_and_degenerate_quotes_fail(self):
        self.assertFalse(verify_quote("management believes the outcome will not have a material adverse effect", CONCENTRATION))
        self.assertFalse(verify_quote("22% of", CONCENTRATION))  # too short to be evidence
        self.assertFalse(verify_quote(None, CONCENTRATION))
        self.assertFalse(verify_quote("sales to one direct customer represented 22%", None))


class TestNumericSupport(unittest.TestCase):
    def test_numbers_must_appear_in_evidence(self):
        self.assertTrue(numbers_supported("Two direct customers at 22% and 14% of revenue", [CONCENTRATION]))
        self.assertFalse(numbers_supported("One customer is 35% of revenue", [CONCENTRATION]))
        self.assertTrue(numbers_supported("A $4,500.0 million charge", ["we incurred a $4500 million charge"]))
        self.assertTrue(numbers_supported(None, [CONCENTRATION]))
        self.assertTrue(numbers_supported("Most revenue comes from few customers", [CONCENTRATION]))


class TestFindQuoteChunk(unittest.TestCase):
    def test_claimed_chunk_accepts_a_near_verbatim_quote(self):
        pool = _pool(("a", UNRELATED), ("b", CONCENTRATION))
        self.assertEqual(find_quote_chunk(NEAR, "b", pool)["chunk_id"], "b")

    def test_other_chunks_accept_only_exact_quotes(self):
        pool = _pool(("a", UNRELATED), ("b", CONCENTRATION))
        self.assertIsNone(find_quote_chunk(NEAR, "a", pool))
        self.assertEqual(find_quote_chunk(EXACT, "a", pool)["chunk_id"], "b")

    def test_unknown_claimed_chunk_falls_back_to_the_pool(self):
        self.assertEqual(find_quote_chunk(EXACT, "missing", _pool(("b", CONCENTRATION)))["chunk_id"], "b")

    def test_first_exact_match_in_pool_order_wins(self):
        pool = _pool(("x", CONCENTRATION), ("y", CONCENTRATION))
        self.assertEqual(find_quote_chunk(EXACT, "", pool)["chunk_id"], "x")

    def test_fabricated_quote_and_empty_pool_find_nothing(self):
        fabricated = "management expects no material adverse effect from these customers at all"
        self.assertIsNone(find_quote_chunk(fabricated, "b", _pool(("b", CONCENTRATION))))
        self.assertIsNone(find_quote_chunk(EXACT, "b", {}))
