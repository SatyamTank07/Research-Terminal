"""Tax quality: effective rates, the statutory fallback and the TAX_ANOMALY screen."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.tax_quality import (
    STATUTORY_TAX_RATE,
    latest_tax_rate,
    measure_tax_quality,
    screen_tax_quality,
    tax_effect_rate,
)


def _year(fy=2025, pretax=100.0, tax=21.0, ni=79.0, deferred=None):
    return AnnualFinancialInput(fiscal_year=fy, revenue=500.0, operating_income=100.0, net_income=ni,
                                operating_cash_flow=90.0, capital_expenditures=10.0, pretax_income=pretax,
                                income_tax_expense=tax, deferred_income_taxes=deferred)


def _screen(year):
    return screen_tax_quality([year], measure_tax_quality([year]))


class TestTaxRates(unittest.TestCase):
    def test_the_latest_rate_is_derived_from_the_10k_when_pre_tax_income_is_taxed(self):
        rate = latest_tax_rate(_year(pretax=100.0, tax=18.0))
        self.assertEqual((rate.rate, rate.source), (0.18, "derived_from_10k"))

    def test_a_pre_tax_loss_falls_back_to_the_statutory_rate(self):
        rate = latest_tax_rate(_year(pretax=-10.0, tax=1.0))
        self.assertEqual((rate.rate, rate.source), (STATUTORY_TAX_RATE, "statutory_fallback_due_to_pretax_loss"))

    def test_a_tax_benefit_falls_back_to_the_statutory_rate(self):
        self.assertEqual(latest_tax_rate(_year(tax=-5.0)).source, "statutory_fallback_due_to_tax_benefit")

    def test_missing_tax_lines_use_the_statutory_default(self):
        self.assertEqual(latest_tax_rate(_year(pretax=None, tax=None)).source, "statutory_default_not_provided")

    def test_adjustments_are_tax_effected_at_the_year_rate_else_statutory(self):
        self.assertEqual(tax_effect_rate(_year(pretax=100.0, tax=30.0)), 0.3)
        self.assertEqual(tax_effect_rate(_year(pretax=100.0, tax=-5.0)), STATUTORY_TAX_RATE)


class TestTaxAnomaly(unittest.TestCase):
    def test_a_tax_benefit_on_pre_tax_profit_is_a_high_anomaly(self):
        findings = _screen(_year(tax=-50.0, ni=150.0))
        self.assertEqual([(f.code, f.severity) for f in findings], [("TAX_ANOMALY", "high")])

    def test_an_effective_rate_outside_5_to_40_percent_is_a_medium_anomaly(self):
        findings = _screen(_year(tax=3.0, ni=97.0))
        self.assertEqual([(f.code, f.severity, f.value) for f in findings], [("TAX_ANOMALY", "medium", 3.0)])

    def test_deferred_tax_above_25_percent_of_net_income_is_an_anomaly(self):
        self.assertEqual(len(_screen(_year(deferred=30.0))), 1)

    def test_a_normal_rate_raises_nothing(self):
        self.assertEqual(_screen(_year()), [])


if __name__ == "__main__":
    unittest.main()
