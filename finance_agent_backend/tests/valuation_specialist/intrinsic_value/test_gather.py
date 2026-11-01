"""intrinsic_value gather: which cash flows are valued and what the bridge reads."""

import unittest
from datetime import date

from app.agents.specialized.valuation_specialist.modules.intrinsic_value.gather import gather_intrinsic_value_inputs
from tests.forecast_fixtures import make_aapl_audit, make_forecast_output
from tests.valuation_fixtures import make_market


class TestIntrinsicValueGather(unittest.TestCase):
    def test_forecast_cash_flows_are_valued_as_projected(self):
        audit = make_aapl_audit()
        forecast = make_forecast_output("AAPL", 2025, audit=audit)
        inputs = gather_intrinsic_value_inputs(audit, forecast, make_market(), date(2026, 10, 11), "2025-09-27")
        self.assertEqual(inputs.projected_fcfs, list(forecast.projected_fcfs))
        self.assertEqual(inputs.forecast_source, "forecasting_analyst")
        self.assertEqual(inputs.cash_flow_warnings, ())
        self.assertEqual(inputs.balance_sheet_date, date(2025, 9, 27))
        self.assertIs(inputs.final_year, forecast.forecast_schedule[-1])

    def test_forecast_without_cash_flows_falls_back_to_placeholder_but_keeps_its_diagnostics(self):
        audit = make_aapl_audit()
        forecast = make_forecast_output("AAPL", 2025, audit=audit).model_copy(update={"projected_fcfs": []})
        inputs = gather_intrinsic_value_inputs(audit, forecast, make_market(), date(2026, 10, 11), None)
        self.assertEqual(inputs.forecast_source, "naive_5pct_fallback")
        self.assertEqual(len(inputs.projected_fcfs), 5)
        self.assertIn("placeholder", inputs.cash_flow_warnings[0])
        self.assertIs(inputs.forecast_diagnostics, forecast.diagnostics)
        self.assertIsNone(inputs.balance_sheet_date)


if __name__ == "__main__":
    unittest.main()
