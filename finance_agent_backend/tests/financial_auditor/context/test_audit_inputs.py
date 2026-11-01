"""Audit inputs: unit and share-scale guards, the contiguous window and the balance-sheet year match."""

import unittest

from pydantic import ValidationError

from app.agents.specialized.financial_auditor.context.audit_inputs import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
    check_audit_window,
    validate_audit_inputs,
)


def _year(fy: int) -> dict:
    return {"fiscal_year": fy, "revenue": 100.0, "operating_income": 20.0, "net_income": 15.0,
            "operating_cash_flow": 25.0, "capital_expenditures": 5.0}


def _balance(fy: int) -> dict:
    return {"fiscal_year": fy, "cash_and_equivalents": 10.0, "stockholders_equity": 50.0,
            "weighted_diluted_shares": 10.0}


class TestInputGuards(unittest.TestCase):
    def test_capex_must_be_a_positive_outflow(self):
        with self.assertRaisesRegex(ValidationError, "positive magnitude"):
            AnnualFinancialInput.model_validate({**_year(2025), "capital_expenditures": -5.0})

    def test_share_counts_in_thousands_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "MILLIONS"):
            BalanceSheetInput.model_validate({**_balance(2025), "weighted_diluted_shares": 15_004_697.0})

    def test_the_latest_balance_sheet_needs_a_share_count(self):
        bs = _balance(2025)
        del bs["weighted_diluted_shares"]
        with self.assertRaisesRegex(ValidationError, "period_end_shares_outstanding"):
            BalanceSheetInput.model_validate(bs)

    def test_missing_securities_and_debt_count_as_zero(self):
        bs = BalanceSheetInput.model_validate({**_balance(2025), "short_term_debt": None})
        self.assertEqual((bs.marketable_securities, bs.short_term_debt, bs.long_term_debt), (0.0, 0.0, 0.0))

    def test_aliases_accept_common_label_spellings(self):
        raw = {k: v for k, v in _year(2025).items() if k != "capital_expenditures"}
        year = AnnualFinancialInput.model_validate({**raw, "capex": 7.0, "sbc": 2.0})
        self.assertEqual((year.capital_expenditures, year.stock_based_compensation), (7.0, 2.0))


class TestAuditWindowGuards(unittest.TestCase):
    def _models(self, years, bs_year, prior_year=None):
        annual = [AnnualFinancialInput.model_validate(_year(y)) for y in years]
        prior = PriorBalanceSheetInput.model_validate(_balance(prior_year)) if prior_year else None
        return annual, BalanceSheetInput.model_validate(_balance(bs_year)), prior

    def test_a_contiguous_window_with_matching_balance_sheets_passes(self):
        check_audit_window(*self._models([2023, 2024, 2025], 2025, 2024))

    def test_an_empty_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one fiscal year"):
            check_audit_window([], BalanceSheetInput.model_validate(_balance(2025)))

    def test_a_gap_in_the_years_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            check_audit_window(*self._models([2022, 2024, 2025], 2025))

    def test_the_balance_sheet_must_sit_at_the_latest_year(self):
        with self.assertRaisesRegex(ValueError, "must match the latest"):
            check_audit_window(*self._models([2024, 2025], 2024))

    def test_the_prior_balance_sheet_must_be_the_year_before(self):
        with self.assertRaisesRegex(ValueError, "the year before the latest balance sheet"):
            check_audit_window(*self._models([2024, 2025], 2025, 2023))


class TestValidateAuditInputs(unittest.TestCase):
    def test_plain_dicts_become_input_models(self):
        annual, bs, prior = validate_audit_inputs({
            "annual_financials": [_year(2024), _year(2025)],
            "balance_sheet": _balance(2025),
            "prior_balance_sheet": _balance(2024),
        })
        self.assertEqual([a.fiscal_year for a in annual], [2024, 2025])
        self.assertIsInstance(bs, BalanceSheetInput)
        self.assertIsInstance(prior, PriorBalanceSheetInput)

    def test_no_prior_balance_sheet_stays_none(self):
        _, _, prior = validate_audit_inputs({"annual_financials": [_year(2025)], "balance_sheet": _balance(2025),
                                             "prior_balance_sheet": None})
        self.assertIsNone(prior)


if __name__ == "__main__":
    unittest.main()
