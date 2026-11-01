"""Revenue quality: receivables growing well ahead of revenue."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history
from app.agents.specialized.financial_auditor.modules.revenue_quality import screen_revenue_quality


def _year(fy, revenue, ar):
    return AnnualFinancialInput(fiscal_year=fy, revenue=revenue, operating_income=20.0, net_income=15.0,
                                operating_cash_flow=25.0, capital_expenditures=5.0, accounts_receivable=ar)


def _screen(years):
    return screen_revenue_quality(years, build_financial_history(years))


class TestRevenueQuality(unittest.TestCase):
    def test_receivables_outgrowing_revenue_by_more_than_15_points_is_a_divergence(self):
        findings = _screen([_year(2024, 100.0, 10.0), _year(2025, 105.0, 13.0)])
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].code, findings[0].fiscal_year, findings[0].value), ("RECEIVABLES_DIVERGENCE", 2025, 25.0))

    def test_small_receivables_growth_is_ignored_even_when_revenue_falls(self):
        self.assertEqual(_screen([_year(2024, 100.0, 10.0), _year(2025, 80.0, 10.4)]), [])

    def test_receivables_tracking_revenue_raise_nothing(self):
        self.assertEqual(_screen([_year(2024, 100.0, 10.0), _year(2025, 120.0, 12.0)]), [])


if __name__ == "__main__":
    unittest.main()
