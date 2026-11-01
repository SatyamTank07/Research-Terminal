"""Financial history: FCF, margins, cash conversion and growth per audited year."""

import math
import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.financial_history import (
    build_financial_history,
    gross_profit_of,
    growth_pct,
    operating_expenses_of,
    pct_of,
    require_finite,
)


def _year(fy, revenue=100.0, ebit=20.0, ni=15.0, ocf=25.0, capex=5.0, **extra):
    return AnnualFinancialInput(fiscal_year=fy, revenue=revenue, operating_income=ebit, net_income=ni,
                                operating_cash_flow=ocf, capital_expenditures=capex, **extra)


class TestBuildFinancialHistory(unittest.TestCase):
    def test_years_come_back_oldest_first_whatever_the_input_order(self):
        history = build_financial_history([_year(2025), _year(2023), _year(2024)])
        self.assertEqual([h.fiscal_year for h in history], [2023, 2024, 2025])

    def test_free_cash_flow_is_operating_cash_flow_minus_capex(self):
        h = build_financial_history([_year(2025, ocf=25.0, capex=5.0, ni=16.0)])[0]
        self.assertEqual(h.free_cash_flow, 20.0)
        self.assertEqual(h.fcf_conversion_pct, 125.0)

    def test_growth_is_measured_against_the_prior_year_magnitude(self):
        history = build_financial_history([_year(2024, ebit=-10.0), _year(2025, ebit=5.0)])
        self.assertIsNone(history[0].revenue_growth_pct)
        self.assertEqual(history[1].operating_income_growth_pct, 150.0)

    def test_gross_profit_is_derived_from_cost_of_revenue_when_not_reported(self):
        h = build_financial_history([_year(2025, cost_of_revenue=60.0)])[0]
        self.assertEqual((h.gross_profit, h.gross_margin_pct), (40.0, 40.0))

    def test_interest_expense_is_reported_as_a_positive_cost(self):
        h = build_financial_history([_year(2025, interest_expense=-7.0)])[0]
        self.assertEqual(h.interest_expense, 7.0)


class TestHistoryHelpers(unittest.TestCase):
    def test_a_ratio_with_a_zero_or_missing_base_is_none(self):
        self.assertIsNone(pct_of(1.0, 0.0))
        self.assertIsNone(pct_of(None, 5.0))
        self.assertIsNone(growth_pct(5.0, 0.0))

    def test_operating_expenses_are_gross_profit_minus_operating_income_when_not_reported(self):
        self.assertEqual(operating_expenses_of(_year(2025, gross_profit=45.0, ebit=20.0)), 25.0)
        self.assertIsNone(operating_expenses_of(_year(2025)))
        self.assertIsNone(gross_profit_of(_year(2025)))

    def test_a_non_finite_line_item_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            require_finite(math.nan, "revenue (2025)")


if __name__ == "__main__":
    unittest.main()
