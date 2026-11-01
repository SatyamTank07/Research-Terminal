"""Working capital: days in receivables, inventory and payables, and the screens on them."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history
from app.agents.specialized.financial_auditor.modules.working_capital import (
    measure_working_capital,
    screen_working_capital,
)


def _year(fy, revenue=365.0, cogs=182.5, ar=None, inv=None, ap=None):
    return AnnualFinancialInput(fiscal_year=fy, revenue=revenue, cost_of_revenue=cogs, operating_income=50.0,
                                net_income=40.0, operating_cash_flow=60.0, capital_expenditures=10.0,
                                accounts_receivable=ar, inventories=inv, accounts_payable=ap)


def _screen(years):
    return screen_working_capital(years, build_financial_history(years), measure_working_capital(years))


class TestWorkingCapitalDays(unittest.TestCase):
    def test_days_use_revenue_for_receivables_and_cost_of_revenue_for_inventory_and_payables(self):
        row = measure_working_capital([_year(2025, ar=30.0, inv=20.0, ap=10.0)])[0]
        self.assertEqual((row.dso_days, row.dio_days, row.dpo_days), (30.0, 40.0, 20.0))
        self.assertEqual(row.cash_conversion_cycle_days, 50.0)

    def test_no_payables_means_no_cash_conversion_cycle(self):
        row = measure_working_capital([_year(2025, ar=30.0, inv=20.0)])[0]
        self.assertIsNone(row.cash_conversion_cycle_days)

    def test_day_changes_compare_with_the_prior_year(self):
        rows = measure_working_capital([_year(2024, ar=30.0), _year(2025, ar=36.0)])
        self.assertIsNone(rows[0].dso_change_pct)
        self.assertEqual(rows[1].dso_change_pct, 20.0)


class TestWorkingCapitalScreens(unittest.TestCase):
    def test_dso_lengthening_more_than_15_percent_is_working_capital_deterioration(self):
        findings = _screen([_year(2024, ar=30.0), _year(2025, ar=36.0)])
        self.assertEqual([(f.code, f.severity, f.metric) for f in findings],
                         [("WORKING_CAPITAL_DETERIORATION", "low", "dso_change_pct")])

    def test_inventory_outgrowing_revenue_by_20_points_is_inventory_accumulation(self):
        findings = _screen([_year(2024, inv=20.0), _year(2025, inv=26.0)])
        self.assertIn("INVENTORY_ACCUMULATION", {f.code for f in findings})

    def test_inventory_growing_with_revenue_raises_nothing(self):
        self.assertEqual(_screen([_year(2024, inv=20.0), _year(2025, revenue=438.0, cogs=219.0, inv=24.0)]), [])


if __name__ == "__main__":
    unittest.main()
