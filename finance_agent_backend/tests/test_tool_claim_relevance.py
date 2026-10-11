"""Claim/quote relevance screen (app/agents/tools/claim_relevance.py), using quotes from a live 10-K run."""

import unittest

from app.agents.tools.claim_relevance import (
    is_ip_boilerplate,
    moat_quote_supports,
    pricing_claim_supported,
    threat_quote_supports,
)

IP_BOILERPLATE = (
    "We regard our trademarks, service marks, copyrights, patents, domain names, trade dress, trade secrets, "
    "proprietary technologies, and similar intellectual property as critical to our success"
)
STRATEGY = "We serve consumers through our online and physical stores and focus on selection, price, and convenience."
LOW_PRICES = "One way we offer lower prices is through shipping offers."


class TestClaimRelevance(unittest.TestCase):
    def test_live_run_off_topic_moat_quotes_are_rejected(self):
        self.assertTrue(is_ip_boilerplate(IP_BOILERPLATE))
        self.assertFalse(moat_quote_supports("Intangible Assets / Brand", IP_BOILERPLATE))
        self.assertFalse(moat_quote_supports("Network Effects", STRATEGY))

    def test_mechanism_quotes_are_accepted(self):
        cases = {
            "Network Effects": "Our marketplace attracts more sellers, which in turn attracts more customers.",
            "High Switching Costs": "Migrating these workloads to another provider requires significant engineering effort.",
            "Cost Advantage": "We operate a global logistics network that lowers our cost to serve each order.",
            "Intangible Assets / Brand": "Our patents cover the core compounds through 2031 and the brand commands customer trust.",
            "Efficient Scale": "There are a limited number of competitors able to serve this market given high capital requirements.",
        }
        for moat, quote in cases.items():
            with self.subTest(moat=moat):
                self.assertTrue(moat_quote_supports(moat, quote))

    def test_unknown_moat_or_empty_quote(self):
        self.assertFalse(moat_quote_supports("Made-up Moat", "network effects everywhere"))
        self.assertFalse(moat_quote_supports(None, "scale"))
        self.assertFalse(moat_quote_supports("Cost Advantage", ""))

    def test_threat_quotes(self):
        self.assertTrue(threat_quote_supports("Some of our current and potential competitors have greater resources."))
        self.assertTrue(threat_quote_supports("The internet and artificial intelligence facilitate competitive entry."))
        self.assertFalse(threat_quote_supports("Our businesses encompass a large variety of product types."))

    def test_pricing_power_claims_need_pricing_power_evidence(self):
        claim = "The company maintains pricing power through its focus on customer experience."
        self.assertFalse(pricing_claim_supported(claim, LOW_PRICES))
        self.assertTrue(pricing_claim_supported(claim, "We implemented price increases across most product lines."))
        # Describing price pressure or a low-price strategy is not a pricing-power claim.
        self.assertTrue(pricing_claim_supported("Prices fell under competitive pressure.", LOW_PRICES))


if __name__ == "__main__":
    unittest.main()
