"""market_implied_expectations calculate: the bounded reverse-DCF root finder."""

import unittest

from app.agents.specialized.valuation_specialist.modules.market_implied_expectations.calculate import solve_reverse_dcf
from app.agents.specialized.valuation_specialist.modules.market_implied_expectations.thresholds import (
    REVERSE_DCF_MAX_ITERATIONS,
    REVERSE_DCF_SCAN_POINTS,
)


class TestSolveReverseDcf(unittest.TestCase):
    def test_linear_root(self):
        root, status = solve_reverse_dcf(lambda g: 100.0 + 1000.0 * g, 150.0)
        self.assertEqual(status, "solved")
        self.assertAlmostEqual(root, 0.05, places=4)

    def test_picks_bracket_nearest_anchor(self):
        def value(g):  # two crossings of 0: at -0.1 and 0.3
            return (g + 0.1) * (g - 0.3)
        self.assertAlmostEqual(solve_reverse_dcf(value, 0.0, anchor=0.25)[0], 0.3, places=4)
        self.assertAlmostEqual(solve_reverse_dcf(value, 0.0, anchor=-0.05)[0], -0.1, places=4)

    def test_no_root_reports_side(self):
        self.assertEqual(solve_reverse_dcf(lambda g: 10.0, 50.0), (None, "price_above_range"))
        self.assertEqual(solve_reverse_dcf(lambda g: 100.0, 50.0), (None, "price_below_range"))

    def test_undefined_or_raising_value_fn(self):
        self.assertEqual(solve_reverse_dcf(lambda g: None, 50.0), (None, "no_solution"))

        def boom(g):
            raise ValueError("bad input")
        self.assertEqual(solve_reverse_dcf(boom, 50.0), (None, "no_solution"))
        self.assertEqual(solve_reverse_dcf(lambda g: float("nan"), 50.0), (None, "no_solution"))

    def test_partially_undefined_curve_still_solves(self):
        root, status = solve_reverse_dcf(lambda g: None if g < 0 else 100.0 + 1000.0 * g, 200.0)
        self.assertEqual(status, "solved")
        self.assertAlmostEqual(root, 0.10, places=4)

    def test_bounded_evaluations(self):
        calls = []

        def value(g):
            calls.append(g)
            return 100.0 + 1000.0 * g
        solve_reverse_dcf(value, 123.456)
        self.assertLessEqual(len(calls), REVERSE_DCF_SCAN_POINTS + REVERSE_DCF_MAX_ITERATIONS)


if __name__ == "__main__":
    unittest.main()
