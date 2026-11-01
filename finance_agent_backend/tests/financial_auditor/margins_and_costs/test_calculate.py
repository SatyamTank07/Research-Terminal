"""Margins and costs: opex leverage and operating-margin compression."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history
from app.agents.specialized.financial_auditor.modules.margins_and_costs import (
    measure_cost_structure,
    screen_margins_and_costs,
)


def _year(fy, revenue, ebit, gross_profit=None, opex=None):
    return AnnualFinancialInput(fiscal_year=fy, revenue=revenue, operating_income=ebit, net_income=ebit * 0.8,
                                operating_cash_flow=ebit, capital_expenditures=1.0, gross_profit=gross_profit,
                                operating_expenses=opex)


def _audit(years):
    history = build_financial_history(years)
    cost = measure_cost_structure(years, history)
    return cost, screen_margins_and_costs(history, cost)


class TestCostStructure(unittest.TestCase):
    def test_margin_moves_are_in_basis_points_against_the_prior_year(self):
        cost, _ = _audit([_year(2024, 100.0, 20.0, 50.0, 30.0), _year(2025, 100.0, 18.0, 48.0, 30.0)])
        self.assertIsNone(cost[0].operating_margin_change_bps)
        self.assertEqual((cost[1].gross_margin_change_bps, cost[1].operating_margin_change_bps), (-200.0, -200.0))

    def test_opex_growth_is_compared_with_revenue_growth_in_points(self):
        cost, _ = _audit([_year(2024, 100.0, 20.0, 50.0, 30.0), _year(2025, 110.0, 20.0, 56.0, 36.0)])
        self.assertEqual(cost[1].opex_growth_minus_revenue_growth_pp, 10.0)


class TestMarginScreens(unittest.TestCase):
    def test_opex_outgrowing_revenue_by_more_than_ten_points_is_negative_operating_leverage(self):
        _, findings = _audit([_year(2024, 100.0, 20.0, 50.0, 30.0), _year(2025, 105.0, 15.0, 51.0, 36.0)])
        self.assertIn("NEGATIVE_OPERATING_LEVERAGE", {f.code for f in findings})

    def test_a_300bps_single_year_drop_is_margin_compression(self):
        _, findings = _audit([_year(2023, 100.0, 20.0), _year(2024, 100.0, 21.0), _year(2025, 100.0, 17.0)])
        compression = [f for f in findings if f.code == "MARGIN_COMPRESSION"]
        self.assertEqual(len(compression), 1)
        self.assertEqual(compression[0].fiscal_year, 2025)
        self.assertEqual(compression[0].value, -300.0)

    def test_steady_margins_raise_nothing(self):
        _, findings = _audit([_year(2024, 100.0, 20.0, 50.0, 30.0), _year(2025, 110.0, 22.0, 55.0, 33.0)])
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
