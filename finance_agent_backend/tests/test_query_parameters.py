"""Deterministic valuation-parameter parsing and session carry-over (app/agents/query_parameters.py)."""

import unittest

from app.agents.query_parameters import (
    SESSION_KEY,
    ValuationOverrides,
    parse_valuation_overrides,
    resolve_session_overrides,
)


class TestParseValuationOverrides(unittest.TestCase):
    def test_terminal_growth_forms(self):
        cases = {
            "DCF for the company with terminal growth 3%": 0.03,
            "terminal growth 0.5%": 0.005,
            "perpetual growth 0.02": 0.02,
            "use a terminal growth rate of 2.75%": 0.0275,
            "perpetuity growth rate = 2": 0.02,
            "assume 3% terminal growth": 0.03,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(parse_valuation_overrides(text).terminal_growth, expected)

    def test_segment_or_revenue_growth_is_not_terminal_growth(self):
        for text in (
            "Run a DCF, cloud segment growth rate 20%",
            "Value it assuming revenue growth rate of 10%",
            "long-term growth rate 12%",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_valuation_overrides(text).terminal_growth)

    def test_discount_rate_forms(self):
        cases = {
            "Calculate DCF with 9% WACC": 0.09,
            "discount rate of 8.5%": 0.085,
            "WACC 0.1": 0.10,
            "hurdle rate is 11": 0.11,
            "cost of capital: 7.5%": 0.075,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(parse_valuation_overrides(text).wacc, expected)

    def test_implausible_values_are_ignored_with_a_note(self):
        out = parse_valuation_overrides("WACC 45% with terminal growth 9% and beta 12")
        self.assertIsNone(out.wacc)
        self.assertIsNone(out.terminal_growth)
        self.assertIsNone(out.beta)
        self.assertEqual(len(out.notes), 3)
        self.assertTrue(all("Ignored" in n for n in out.notes))

    def test_terminal_growth_must_stay_below_wacc(self):
        out = parse_valuation_overrides("WACC 4% and terminal growth 4.5%")
        self.assertAlmostEqual(out.wacc, 0.04)
        self.assertIsNone(out.terminal_growth)
        self.assertIn("below the 4.00% discount rate", out.notes[0])

    def test_beta_and_price(self):
        out = parse_valuation_overrides("value it with beta 1.3, share price $1,250.50")
        self.assertAlmostEqual(out.beta, 1.3)
        self.assertAlmostEqual(out.share_price, 1250.5)
        self.assertAlmostEqual(parse_valuation_overrides("trading at 42").share_price, 42.0)

    def test_years_and_price_targets_are_not_prices(self):
        for text in ("what was the price 2024 close", "price target of 300", "price-to-earnings 25"):
            with self.subTest(text=text):
                self.assertIsNone(parse_valuation_overrides(text).share_price)
        self.assertAlmostEqual(parse_valuation_overrides("price $2024").share_price, 2024.0)

    def test_no_parameters(self):
        out = parse_valuation_overrides("Analyze the company")
        self.assertEqual(out.values(), {})
        self.assertFalse(out.reset)
        self.assertEqual(parse_valuation_overrides(None).values(), {})

    def test_reset_detection(self):
        for text in ("reset the assumptions", "use CAPM again", "clear my WACC override", "back to default"):
            with self.subTest(text=text):
                self.assertTrue(parse_valuation_overrides(text).reset)


class TestResolveSessionOverrides(unittest.TestCase):
    def test_unbound_values_attach_to_the_next_ticker(self):
        pending = resolve_session_overrides(parse_valuation_overrides("Run a DCF at 8% WACC"), None, None)
        self.assertEqual(pending, {"ticker": None, "wacc": 0.08})
        bound = resolve_session_overrides(ValuationOverrides(), pending, "abc")
        self.assertEqual(bound, {"ticker": "ABC", "wacc": 0.08})

    def test_same_ticker_carries_and_current_values_win(self):
        stored = {"ticker": "ABC", "wacc": 0.08, "beta": 1.2}
        merged = resolve_session_overrides(parse_valuation_overrides("now with WACC 9%"), stored, "ABC")
        self.assertEqual(merged, {"ticker": "ABC", "wacc": 0.09, "beta": 1.2})

    def test_switching_company_drops_remembered_values(self):
        stored = {"ticker": "ABC", "wacc": 0.08}
        self.assertEqual(resolve_session_overrides(ValuationOverrides(), stored, "XYZ"), {})
        merged = resolve_session_overrides(parse_valuation_overrides("beta 0.9"), stored, "XYZ")
        self.assertEqual(merged, {"ticker": "XYZ", "beta": 0.9})

    def test_reset_drops_remembered_values(self):
        stored = {"ticker": "ABC", "wacc": 0.08}
        self.assertEqual(resolve_session_overrides(parse_valuation_overrides("use CAPM"), stored, "ABC"), {})

    def test_session_key_name_is_stable(self):
        self.assertEqual(SESSION_KEY, "valuation_overrides")


if __name__ == "__main__":
    unittest.main()
