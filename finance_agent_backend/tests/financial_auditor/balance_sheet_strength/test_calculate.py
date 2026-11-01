"""Balance-sheet strength: liquidity bridge, valuation share count, solvency ratios and negative book equity."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
)
from app.agents.specialized.financial_auditor.modules.balance_sheet_strength import (
    classify_book_equity,
    measure_balance_sheet,
    measure_solvency,
    screen_balance_sheet_strength,
)
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history


def _bs(equity=100.0, wd=10.0, pe=None, **extra):
    return BalanceSheetInput(fiscal_year=2025, cash_and_equivalents=30.0, marketable_securities=20.0,
                             short_term_debt=10.0, long_term_debt=60.0, stockholders_equity=equity,
                             weighted_diluted_shares=wd, period_end_shares_outstanding=pe, **extra)


def _latest(da=10.0):
    year = AnnualFinancialInput(fiscal_year=2025, revenue=500.0, operating_income=40.0, net_income=30.0,
                                operating_cash_flow=50.0, capital_expenditures=10.0, depreciation_amortization=da)
    return build_financial_history([year])[0], year


class TestBalanceSheet(unittest.TestCase):
    def test_net_debt_is_gross_debt_minus_cash_and_marketable_securities(self):
        bs = measure_balance_sheet(_bs())
        self.assertEqual((bs.total_liquid_cash, bs.total_debt, bs.net_debt), (50.0, 70.0, 20.0))
        self.assertFalse(bs.net_cash_position)

    def test_the_valuation_share_count_never_drops_below_period_end_shares(self):
        bs = measure_balance_sheet(_bs(wd=10.0, pe=12.0))
        self.assertEqual((bs.valuation_shares_outstanding, bs.share_count_source),
                         (12.0, "period_end_basic_exceeds_weighted_diluted"))
        self.assertEqual(measure_balance_sheet(_bs(wd=None, pe=9.0)).share_count_source, "period_end_basic_only")

    def test_prior_gross_debt_is_kept_for_the_average_debt_cost(self):
        prior = PriorBalanceSheetInput(fiscal_year=2024, cash_and_equivalents=1.0, short_term_debt=5.0,
                                       long_term_debt=45.0, stockholders_equity=90.0)
        self.assertEqual(measure_balance_sheet(_bs(), prior).prior_total_debt, 50.0)


class TestSolvency(unittest.TestCase):
    def test_leverage_is_net_debt_over_ebitda(self):
        latest, year = _latest(da=10.0)
        balance = measure_balance_sheet(_bs())
        solvency = measure_solvency(balance, latest, year, classify_book_equity(100.0))
        self.assertEqual((solvency.ebitda, solvency.net_debt_to_ebitda), (50.0, 0.4))
        self.assertEqual(solvency.net_debt_to_ebitda_interpretation, "leverage_ratio")
        self.assertEqual(solvency.total_debt_to_equity, 0.7)

    def test_debt_to_equity_is_suppressed_on_negative_book_equity(self):
        latest, year = _latest()
        balance = measure_balance_sheet(_bs(equity=-5.0))
        status = classify_book_equity(-5.0)
        self.assertEqual(status, "negative_book_equity_ratios_suppressed")
        self.assertIsNone(measure_solvency(balance, latest, year, status).total_debt_to_equity)

    def test_without_d_and_a_there_is_no_ebitda(self):
        latest, year = _latest(da=None)
        solvency = measure_solvency(measure_balance_sheet(_bs()), latest, year, "positive_book_equity")
        self.assertEqual(solvency.net_debt_to_ebitda_interpretation, "ebitda_not_available")


class TestNegativeBookEquity(unittest.TestCase):
    def test_negative_equity_is_a_high_finding(self):
        findings = screen_balance_sheet_strength(_bs(equity=-5.0))
        self.assertEqual([(f.code, f.severity) for f in findings], [("NEGATIVE_BOOK_EQUITY", "high")])

    def test_positive_equity_raises_nothing(self):
        self.assertEqual(screen_balance_sheet_strength(_bs()), [])
        self.assertEqual(classify_book_equity(0.0), "zero_book_equity_ratios_suppressed")


if __name__ == "__main__":
    unittest.main()
