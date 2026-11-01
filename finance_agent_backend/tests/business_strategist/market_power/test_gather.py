"""Customer-concentration and pricing sentences screened from outside Item 1."""

import unittest
from unittest.mock import patch

from app.agents.specialized.business_strategist.modules.market_power.gather import find_disclosure_sentences, gather_disclosures
from tests.business_strategist.filing_fixtures import text


class TestSentenceScreens(unittest.TestCase):
    def test_sentence_screens(self):
        chunks = [
            text(1, "Sales to one customer represented 22% of total revenue in fiscal 2025. Prices rose 5%.", item="Item 7"),
            text(2, "As of year end, no entity represented 10% or more of our total receivables balance.", item="Item 8"),
            text(3, "We purchase components from suppliers at competitive pricing to lower our costs.", item="Item 1A"),
            text(4, "We reduced average selling prices on our older products due to competitive pressure.", item="Item 7"),
            text(5, "The tax credit ranges from 6% to 50% for eligible customers installing systems.", item="Item 1"),
        ]
        found = find_disclosure_sentences(chunks, {}, exclude_chunk_ids={"n5"})
        kinds = [(d.kind, d.chunk_id) for d in found]
        self.assertEqual(kinds[0], ("revenue_concentration", "n1"))
        self.assertIn(("receivables_concentration", "n2"), kinds)
        self.assertIn(("pricing", "n4"), kinds)
        self.assertNotIn(("pricing", "n3"), kinds)  # input-cost sentence
        self.assertFalse(any(cid == "n5" for _, cid in kinds))

    def test_a_failing_screen_returns_nothing(self):
        with patch("app.agents.specialized.business_strategist.modules.market_power.gather.find_disclosure_sentences", side_effect=RuntimeError("bug")):
            self.assertEqual(gather_disclosures([text(1, "x")], {}, set()), [])


if __name__ == "__main__":
    unittest.main()
