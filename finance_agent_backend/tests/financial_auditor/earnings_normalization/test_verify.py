"""One-off screen: only discrete, non-recurring items in the audit window are accepted."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.earnings_normalization import (
    NormalizationAdjustment,
    screen_normalization_adjustments,
)

YEAR = AnnualFinancialInput(fiscal_year=2025, revenue=500.0, operating_income=100.0, net_income=88.0,
                            operating_cash_flow=90.0, capital_expenditures=10.0, pretax_income=110.0,
                            income_tax_expense=22.0)


def _screen(label, amount=5.0, fy=2025, rationale=None):
    adj = NormalizationAdjustment(fiscal_year=fy, label=label, amount=amount, direction="depressed_reported_earnings",
                                  rationale=rationale)
    return screen_normalization_adjustments([YEAR], [adj])


class TestScreenNormalizationAdjustments(unittest.TestCase):
    def test_a_discrete_charge_is_accepted(self):
        self.assertEqual(len(_screen("Goodwill impairment")["accepted"]), 1)

    def test_a_year_outside_the_window_is_rejected(self):
        self.assertIn("outside the audited window", _screen("Impairment", fy=2019)["rejected"][0]["rejection"])

    def test_aggregate_other_income_is_rejected(self):
        self.assertIn("aggregate non-operating", _screen("Other income, net")["rejected"][0]["rejection"])

    def test_recurring_operating_costs_are_rejected(self):
        self.assertIn("recurring operating cost", _screen("Stock-based compensation")["rejected"][0]["rejection"])

    def test_the_recurring_tax_provision_is_rejected_but_a_valuation_allowance_release_passes(self):
        self.assertEqual(len(_screen("Income tax expense")["rejected"]), 1)
        self.assertEqual(len(_screen("Tax benefit", rationale="valuation allowance release")["accepted"]), 1)

    def test_an_amount_equal_to_the_whole_non_operating_subtotal_is_rejected(self):
        self.assertIn("non-operating income subtotal", _screen("Gain on sale", amount=10.0)["rejected"][0]["rejection"])


if __name__ == "__main__":
    unittest.main()
