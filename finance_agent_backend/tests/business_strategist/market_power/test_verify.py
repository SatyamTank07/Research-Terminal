"""Pricing power and customer concentration kept only where verified quotes support them."""

import unittest

from app.agents.specialized.business_strategist.modules.market_power import NOT_DISCLOSED, DisclosureSentence
from app.agents.specialized.business_strategist.modules.market_power.verify import is_not_disclosed, verify_market_power

PRICING = "We reduced prices on our entry-level subscription plans in 2025 in response to competitive pressure."
RAISED = "We implemented price increases across our premium tier in 2025."
CONCENTRATION = "No single customer accounted for 10% or more of our net sales in 2025, 2024 or 2023."
RECEIVABLES = "No entity represented 10% or more of our total receivables balance."
POOL = {
    "m1": {"chunk_id": "m1", "item": "Item 7", "breadcrumb": "MD&A", "content": CONCENTRATION},
    "m2": {"chunk_id": "m2", "item": "Item 7", "breadcrumb": "MD&A", "content": PRICING},
    "m3": {"chunk_id": "m3", "item": "Item 7", "breadcrumb": "MD&A", "content": RAISED},
    "n8": {"chunk_id": "n8", "item": "Item 8", "breadcrumb": "Notes", "content": RECEIVABLES},
}
SCREENED = [
    DisclosureSentence(chunk_id="n8", item="Item 8", text=RECEIVABLES, kind="receivables_concentration"),
    DisclosureSentence(chunk_id="m1", item="Item 7", text=CONCENTRATION, kind="revenue_concentration"),
]


def _claim(key, text, chunk_id=None, quote=None):
    field = "assessment" if key == "pricing_power" else "disclosure"
    return {key: {field: text, "evidence": [{"chunk_id": chunk_id, "quote": quote}] if quote else []}}


def _run(data, disclosures=(), assessed=True):
    return verify_market_power(data, POOL, list(disclosures), assessed, "the qualitative LLM call failed")


class TestPricing(unittest.TestCase):
    def test_price_pressure_is_kept(self):
        out = _run(_claim("pricing_power", "The company cut entry-level prices.", "m2", "We reduced prices on our entry-level"))
        self.assertEqual(out.pricing_power_assessment, "The company cut entry-level prices.")
        self.assertEqual([e.claim for e in out.evidence], ["pricing_power"])

    def test_a_pricing_power_claim_needs_a_price_increase_quote(self):
        out = _run(_claim("pricing_power", "The company has pricing power.", "m2", "We reduced prices on our entry-level"))
        self.assertTrue(out.pricing_power_assessment.startswith('Filing states: "We reduced prices'))
        kept = _run(_claim("pricing_power", "The company has pricing power.", "m3", "We implemented price increases"))
        self.assertEqual(kept.pricing_power_assessment, "The company has pricing power.")

    def test_figures_must_be_in_the_quoted_chunk(self):
        out = _run(_claim("pricing_power", "Prices were cut by 15%.", "m2", "We reduced prices on our entry-level"))
        self.assertTrue(out.pricing_power_assessment.startswith("Filing states:"))
        self.assertIn("figures absent", out.tally.notes[0])

    def test_unverified_or_undisclosed_pricing(self):
        self.assertEqual(_run(_claim("pricing_power", "Strong pricing power.", "m2", "we raise prices at will"))
                         .pricing_power_assessment, NOT_DISCLOSED)
        self.assertEqual(_run(_claim("pricing_power", "Not explicitly disclosed in 10-K")).pricing_power_assessment, NOT_DISCLOSED)
        self.assertEqual(_run({}, assessed=False).pricing_power_assessment, "Not assessed: the qualitative LLM call failed.")


class TestConcentration(unittest.TestCase):
    def test_verified_disclosure_is_kept(self):
        out = _run(_claim("customer_concentration", "No customer exceeded 10% of net sales.", "m1",
                          "No single customer accounted for 10% or more of our net sales"))
        self.assertEqual(out.customer_concentration, "No customer exceeded 10% of net sales.")

    def test_unsupported_disclosure_falls_back_to_the_revenue_sentence(self):
        out = _run(_claim("customer_concentration", "One customer is 40% of sales.", "m1",
                          "No single customer accounted for 10% or more of our net sales"), SCREENED)
        self.assertEqual(out.customer_concentration, f'Filing states: "{CONCENTRATION}"')
        self.assertEqual((out.evidence[0].subject, out.evidence[0].chunk_id), ("code_screened", "m1"))

    def test_receivables_only_fallback(self):
        out = _run({}, SCREENED[:1])
        self.assertEqual(out.customer_concentration, f'{NOT_DISCLOSED} for revenue. Receivables: "{RECEIVABLES}"')

    def test_nothing_to_fall_back_on(self):
        self.assertEqual(_run({}).customer_concentration, NOT_DISCLOSED)
        self.assertEqual(_run({}, assessed=False).customer_concentration, "Not assessed: the qualitative LLM call failed.")

    def test_not_disclosed_hints(self):
        self.assertTrue(is_not_disclosed("Not discussed in the filing"))
        self.assertFalse(is_not_disclosed("One customer is 22% of revenue"))


if __name__ == "__main__":
    unittest.main()
