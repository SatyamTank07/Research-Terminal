"""Earnings normalization: tax-effected one-offs, non-operating income and NON_CORE_INCOME_DEPENDENCE."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.earnings_normalization import (
    NormalizationAdjustment,
    normalize_earnings,
    screen_earnings_normalization,
)
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history


def _year(fy=2025, ebit=100.0, pretax=110.0, tax=22.0, ni=88.0):
    return AnnualFinancialInput(fiscal_year=fy, revenue=500.0, operating_income=ebit, net_income=ni,
                                operating_cash_flow=90.0, capital_expenditures=10.0, pretax_income=pretax,
                                income_tax_expense=tax)


def _adj(amount, direction="depressed_reported_earnings", affects="operating_income", fy=2025):
    return NormalizationAdjustment(fiscal_year=fy, label="Restructuring charge", amount=amount,
                                   direction=direction, affects=affects)


def _normalize(years, adjustments=None):
    return normalize_earnings(years, build_financial_history(years), adjustments)


class TestNormalizeEarnings(unittest.TestCase):
    def test_a_pre_tax_charge_is_added_back_to_ebit_and_tax_effected_into_net_income(self):
        row = _normalize([_year()], [_adj(10.0)])[0]
        self.assertEqual(row.normalized_operating_income, 110.0)
        self.assertEqual(row.net_income_adjustments_total, -8.0)  # 10 x (1 - 22/110)
        self.assertEqual(row.normalized_net_income, 96.0)

    def test_a_below_the_line_gain_comes_out_of_net_income_untaxed(self):
        row = _normalize([_year()], [_adj(5.0, "inflated_reported_earnings", "net_income")])[0]
        self.assertEqual((row.normalized_operating_income, row.normalized_net_income), (100.0, 83.0))

    def test_non_operating_income_is_pre_tax_income_minus_operating_income(self):
        row = _normalize([_year()])[0]
        self.assertEqual((row.non_operating_income, row.non_operating_to_operating_income_pct), (10.0, 10.0))

    def test_an_adjustment_outside_the_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the audited window"):
            _normalize([_year()], [_adj(1.0, fy=2019)])


class TestNonCoreIncomeDependence(unittest.TestCase):
    def test_non_operating_income_above_20_percent_of_ebit_is_medium(self):
        findings = screen_earnings_normalization(_normalize([_year(pretax=125.0, ni=100.0)]))
        self.assertEqual([(f.code, f.severity) for f in findings], [("NON_CORE_INCOME_DEPENDENCE", "medium")])

    def test_one_offs_above_20_percent_of_ebit_are_high(self):
        findings = screen_earnings_normalization(_normalize([_year()], [_adj(25.0)]))
        self.assertEqual([(f.severity, f.metric) for f in findings],
                         [("high", "normalization_adjustments_pct_of_operating_income")])


if __name__ == "__main__":
    unittest.main()
