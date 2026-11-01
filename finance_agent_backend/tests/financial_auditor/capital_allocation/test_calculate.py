"""Capital allocation: ROIC / ROE on average capital, capex intensity, dilution and the screens on them."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
)
from app.agents.specialized.financial_auditor.modules.balance_sheet_strength import measure_balance_sheet
from app.agents.specialized.financial_auditor.modules.capital_allocation import (
    measure_capital_intensity,
    measure_capital_returns,
    measure_share_count,
    screen_capital_allocation,
)
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history


def _year(fy, ocf=50.0, capex=10.0, da=8.0, diluted=None, period_end=None):
    return AnnualFinancialInput(fiscal_year=fy, revenue=500.0, operating_income=40.0, net_income=30.0,
                                operating_cash_flow=ocf, capital_expenditures=capex, depreciation_amortization=da,
                                diluted_weighted_shares=diluted, period_end_shares_outstanding=period_end)


BALANCE = BalanceSheetInput(fiscal_year=2025, cash_and_equivalents=20.0, short_term_debt=0.0, long_term_debt=80.0,
                            stockholders_equity=140.0, weighted_diluted_shares=10.0)
PRIOR = PriorBalanceSheetInput(fiscal_year=2024, cash_and_equivalents=20.0, long_term_debt=80.0,
                               stockholders_equity=100.0)


def _screen(years):
    history = build_financial_history(years)
    return screen_capital_allocation(history, measure_capital_intensity(years, history), measure_share_count(years))


class TestCapitalReturns(unittest.TestCase):
    def test_roic_uses_nopat_over_average_invested_capital_net_of_cash(self):
        latest = build_financial_history([_year(2025)])[0]
        returns = measure_capital_returns(latest, measure_balance_sheet(BALANCE), PRIOR, 0.25, "positive_book_equity")
        self.assertEqual((returns.nopat, returns.invested_capital, returns.capital_basis), (30.0, 180.0, "average"))
        self.assertEqual((returns.roic_pct, returns.roe_pct), (16.67, 25.0))

    def test_without_a_prior_balance_sheet_returns_use_ending_capital(self):
        latest = build_financial_history([_year(2025)])[0]
        returns = measure_capital_returns(latest, measure_balance_sheet(BALANCE), None, 0.25, "positive_book_equity")
        self.assertEqual((returns.capital_basis, returns.invested_capital), ("ending", 200.0))

    def test_returns_are_suppressed_without_positive_book_equity(self):
        latest = build_financial_history([_year(2025)])[0]
        returns = measure_capital_returns(latest, measure_balance_sheet(BALANCE), PRIOR, 0.25,
                                          "negative_book_equity_ratios_suppressed")
        self.assertEqual((returns.roic_pct, returns.roic_on_gross_capital_pct, returns.roe_pct), (None, None, None))


class TestCapitalAllocationScreens(unittest.TestCase):
    def test_fcf_rising_only_because_capex_fell_is_capex_driven(self):
        findings = _screen([_year(2024, ocf=50.0, capex=20.0), _year(2025, ocf=50.0, capex=10.0)])
        self.assertIn("CAPEX_DRIVEN_FCF", {f.code for f in findings})

    def test_capex_below_depreciation_two_years_running_is_underinvestment(self):
        findings = _screen([_year(2024, capex=6.0), _year(2025, capex=6.0)])
        self.assertEqual([(f.code, f.severity) for f in findings], [("UNDERINVESTMENT", "info")])

    def test_diluted_share_growth_above_3_percent_is_medium_dilution(self):
        findings = _screen([_year(2024, diluted=100.0), _year(2025, diluted=104.0)])
        self.assertEqual([(f.code, f.severity) for f in findings], [("SHARE_DILUTION", "medium")])

    def test_period_end_share_growth_above_5_percent_is_high_dilution(self):
        findings = _screen([_year(2024, period_end=100.0), _year(2025, period_end=106.0)])
        self.assertEqual([(f.code, f.severity) for f in findings], [("SHARE_DILUTION", "high")])


if __name__ == "__main__":
    unittest.main()
