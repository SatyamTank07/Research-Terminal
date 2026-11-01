"""The evidence rule every judged claim goes through: real quotes only, on topic, at most two per claim."""

import unittest

from app.agents.specialized.business_strategist.evidence import (
    MAX_EVIDENCE_PER_CLAIM,
    MAX_QUOTE_CHARS,
    VerifyTally,
    not_assessed,
    text_of,
    verify_quotes,
)

POOL = {
    "a": {"chunk_id": "a", "item": "Item 1", "breadcrumb": "Business",
          "content": "Each additional seller attracts more buyers to our marketplace. We compete with large retailers."},
    "b": {"chunk_id": "b", "item": "Item 7", "breadcrumb": "MD&A",
          "content": "Migrating workloads to another provider requires significant engineering effort."},
}


def _ev(chunk_id, quote):
    return {"chunk_id": chunk_id, "quote": quote}


class TestVerifyQuotes(unittest.TestCase):
    def test_exact_quote_is_kept_with_its_chunk_item(self):
        tally = VerifyTally()
        kept = verify_quotes([_ev("a", "Each additional seller attracts more buyers")], POOL, tally, "primary_moat", "X")
        self.assertEqual([(e.chunk_id, e.item, e.breadcrumb, e.claim, e.subject) for e in kept],
                         [("a", "Item 1", "Business", "primary_moat", "X")])
        self.assertEqual(tally.quotes_verified, 1)

    def test_near_verbatim_only_on_the_claimed_chunk(self):
        quote = "Each additional seller attracts more buyers to the marketplace"
        self.assertEqual(len(verify_quotes([_ev("a", quote)], POOL, VerifyTally(), "primary_moat", None)), 1)
        tally = VerifyTally()
        self.assertEqual(verify_quotes([_ev("b", quote)], POOL, tally, "primary_moat", None), [])
        self.assertEqual(tally.quotes_rejected, 1)

    def test_exact_quote_under_the_wrong_chunk_is_reattributed(self):
        kept = verify_quotes([_ev("a", "Migrating workloads to another provider")], POOL, VerifyTally(), "secondary_moat", None)
        self.assertEqual(kept[0].chunk_id, "b")

    def test_off_topic_quote_is_counted_not_kept(self):
        tally = VerifyTally()
        kept = verify_quotes([_ev("a", "We compete with large retailers.")], POOL, tally, "primary_moat", None,
                             supports=lambda quote, chunk: "seller" in quote)
        self.assertEqual(kept, [])
        self.assertEqual((tally.quotes_off_topic, tally.quotes_verified), (1, 0))

    def test_duplicates_are_kept_once_and_extra_quotes_ignored(self):
        quote = "Each additional seller attracts more buyers"
        raw = [_ev("a", quote), _ev("a", quote), _ev("b", "Migrating workloads to another provider")]
        kept = verify_quotes(raw, POOL, VerifyTally(), "primary_moat", None)
        self.assertEqual(MAX_EVIDENCE_PER_CLAIM, 2)
        self.assertEqual(len(kept), 1)  # the third quote is past the per-claim limit

    def test_long_quotes_are_cut_to_the_limit(self):
        sentence = " ".join(f"word{i}" for i in range(80))
        pool = {"c": {"chunk_id": "c", "content": sentence}}
        kept = verify_quotes([_ev("c", sentence)], pool, VerifyTally(), "primary_moat", None)
        self.assertEqual(kept[0].quote, sentence[:MAX_QUOTE_CHARS])

    def test_malformed_evidence_is_tolerated(self):
        tally = VerifyTally()
        self.assertEqual(verify_quotes("text", POOL, tally, "primary_moat", None), [])
        self.assertEqual(verify_quotes([None, {"quote": ""}, {"quote": 3}], POOL, tally, "primary_moat", None), [])
        single = verify_quotes(_ev("a", "Each additional seller attracts more buyers"), POOL, tally, "primary_moat", None)
        self.assertEqual(len(single), 1)
        self.assertEqual((tally.quotes_rejected, tally.quotes_off_topic), (0, 0))


class TestTextHelpers(unittest.TestCase):
    def test_text_of(self):
        self.assertEqual(text_of("  moat  "), "moat")
        self.assertEqual(text_of("   "), "")
        self.assertEqual(text_of(None), "")
        self.assertEqual(text_of(["moat"]), "")

    def test_not_assessed(self):
        self.assertEqual(not_assessed("the qualitative LLM call failed"), "Not assessed: the qualitative LLM call failed.")


if __name__ == "__main__":
    unittest.main()
