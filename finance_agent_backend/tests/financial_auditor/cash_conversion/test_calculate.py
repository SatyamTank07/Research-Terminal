"""Cash conversion: accruals, the Sloan ratio, FCF conversion and stock-based compensation."""

import unittest

from app.agents.specialized.financial_auditor.context.audit_inputs import AnnualFinancialInput
from app.agents.specialized.financial_auditor.modules.cash_conversion import (
    measure_cash_conversion,
    screen_cash_conversion,
)
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history


def _year(fy, ni=15.0, ocf=25.0, capex=5.0, ebit=20.0, sbc=None, total_assets=None, ni_total=None):
    return AnnualFinancialInput(fiscal_year=fy, revenue=100.0, operating_income=ebit, net_income=ni,
                                operating_cash_flow=ocf, capital_expenditures=capex, stock_based_compensation=sbc,
                                total_assets=total_assets, net_income_total=ni_total)


def _audit(years, prior_total_assets=None):
    history = build_financial_history(years)
    rows = measure_cash_conversion(years, history, prior_total_assets)
    return rows, screen_cash_conversion(history, rows)


class TestCashConversion(unittest.TestCase):
    def test_accruals_use_consolidated_net_income_and_average_total_assets(self):
        rows, _ = _audit([_year(2025, ni=15.0, ni_total=17.0, ocf=5.0, total_assets=110.0)], prior_total_assets=90.0)
        self.assertEqual(rows[0].accruals, 12.0)
        self.assertEqual(rows[0].sloan_accruals_ratio_pct, 12.0)

    def test_sbc_adjusted_fcf_subtracts_sbc_from_fcf(self):
        rows, _ = _audit([_year(2025, ocf=25.0, capex=5.0, sbc=4.0)])
        self.assertEqual((rows[0].sbc_adjusted_fcf, rows[0].sbc_to_ocf_pct), (16.0, 16.0))

    def test_sbc_ratios_are_suppressed_when_the_base_is_not_positive(self):
        rows, _ = _audit([_year(2025, ocf=-5.0, ebit=-2.0, sbc=4.0)])
        self.assertIsNone(rows[0].sbc_to_ocf_pct)
        self.assertIsNone(rows[0].sbc_to_operating_income_pct)


class TestCashConversionScreens(unittest.TestCase):
    def test_net_income_up_while_ocf_falls_is_an_earnings_quality_divergence(self):
        _, findings = _audit([_year(2024, ni=10.0, ocf=30.0), _year(2025, ni=12.0, ocf=25.0)])
        self.assertIn(("EARNINGS_QUALITY_DIVERGENCE", "high"), {(f.code, f.severity) for f in findings})

    def test_ocf_below_net_income_every_year_is_a_chronic_accrual_deficit(self):
        _, findings = _audit([_year(2024, ni=20.0, ocf=15.0, capex=1.0), _year(2025, ni=20.0, ocf=15.0, capex=1.0)])
        self.assertIn("CHRONIC_ACCRUAL_DEFICIT", {f.code for f in findings})

    def test_fcf_below_70_percent_of_profits_every_year_is_weak_cash_conversion(self):
        _, findings = _audit([_year(2024, ni=20.0, ocf=25.0, capex=12.0), _year(2025, ni=20.0, ocf=25.0, capex=12.0)])
        weak = [f for f in findings if f.code == "WEAK_CASH_CONVERSION"]
        self.assertEqual(weak[0].value, 65.0)

    def test_sbc_above_15_percent_of_ocf_is_sbc_intensity(self):
        _, findings = _audit([_year(2025, sbc=4.0)])
        self.assertEqual([f.code for f in findings], ["SBC_INTENSITY"])

    def test_a_sloan_ratio_above_10_percent_is_high_accruals(self):
        _, findings = _audit([_year(2025, ni=15.0, ocf=4.0, capex=1.0, total_assets=100.0)])
        self.assertIn("HIGH_ACCRUALS", {f.code for f in findings})

    def test_cash_backed_profits_raise_nothing(self):
        self.assertEqual(_audit([_year(2024), _year(2025)])[1], [])


if __name__ == "__main__":
    unittest.main()
