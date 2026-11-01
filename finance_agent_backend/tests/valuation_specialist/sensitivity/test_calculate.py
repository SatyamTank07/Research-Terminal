"""sensitivity calculate: the WACC x growth and WACC x terminal RONIC grids around the base case."""

import unittest

from app.agents.specialized.valuation_specialist.modules.intrinsic_value.calculate import calculate_dcf
from app.agents.specialized.valuation_specialist.modules.sensitivity.calculate import build_sensitivity_grids


class TestSensitivityGrids(unittest.TestCase):
    FCFS = [100.0, 110.0, 120.0, 130.0, 140.0]

    def test_04_sensitivity_matrix_structure(self):
        """Verify 5x5 sensitivity table Markdown output formatting and N/A handling."""
        fcfs = [500.0, 525.0, 550.0, 575.0, 600.0]
        wacc = 0.035
        g = 0.025
        net_debt = 100.0
        shares = 50.0

        grids = build_sensitivity_grids(calculate_dcf(fcfs, wacc, g, net_debt, shares))
        matrix = grids.sensitivity_matrix_markdown

        self.assertIn("| WACC \\ Growth |", matrix)
        self.assertIn("*(Base)*", matrix)
        lines = [line for line in matrix.split("\n") if line.strip()]
        self.assertEqual(len(lines), 7)

        # For wacc - 0.010 (2.5%) and growth + 0.006 (3.1%), w <= g => must contain N/A
        self.assertIn("N/A", matrix)

    def test_base_cell_equals_headline(self):
        res = calculate_dcf(
            [100.0, 110.0, 120.0, 130.0, 140.0], 0.0937, 0.025, -20.0, 10.0,
            terminal_nopat=150.0, ronic=0.25, noncontrolling_interests=5.0, roll_forward_years=0.3,
            risk_free_rate=0.04, terminal_ebitda=180.0,
        )
        self.assertIn(f"**${res.fair_value_per_share:.2f}**", build_sensitivity_grids(res).sensitivity_matrix_markdown)

    def test_ronic_grid_is_informative_when_ronic_equals_wacc(self):
        res = calculate_dcf(self.FCFS, 0.10, 0.025, 0.0, 10.0, terminal_nopat=180.0, ronic=0.10)
        growth_grid = build_sensitivity_grids(res).sensitivity_matrix_markdown
        self.assertIn("growth beyond the horizon creates no value", growth_grid)
        base_row = next(line for line in growth_grid.splitlines() if "*(Base)*" in line)
        cells = [float(c.strip().strip("*$")) for c in base_row.split("|")[2:-1]]
        self.assertLess(max(cells) / min(cells), 1.01)  # TV = NOPAT x (1 + g) / WACC: nearly flat in g

        grid = build_sensitivity_grids(res).ronic_sensitivity_matrix_markdown
        self.assertIn("WACC \\ RONIC", grid)
        self.assertIn(f"**${res.fair_value_per_share:.2f}**", grid)
        ronic_row = next(line for line in grid.splitlines() if "*(Base)*" in line)
        values = [float(c.strip().strip("*$")) for c in ronic_row.split("|")[2:-1]]
        self.assertEqual(values, sorted(values))  # higher RONIC, higher value
        self.assertGreater(values[-1], values[0])

    def test_ronic_grid_columns_floor_above_growth(self):
        res = calculate_dcf(self.FCFS, 0.10, 0.025, 0.0, 10.0, terminal_nopat=180.0, ronic=0.04)
        header = build_sensitivity_grids(res).ronic_sensitivity_matrix_markdown.splitlines()[0]
        self.assertIn("3.5%", header)  # 4% - 4 pp floored at g + 1 pp
        self.assertNotIn("N/A", build_sensitivity_grids(res).ronic_sensitivity_matrix_markdown)

    def test_no_ronic_grid_without_value_driver(self):
        res = calculate_dcf(self.FCFS, 0.10, 0.025, 0.0, 10.0)
        self.assertEqual(build_sensitivity_grids(res).ronic_sensitivity_matrix_markdown, "")
        self.assertNotIn("creates no value", build_sensitivity_grids(res).sensitivity_matrix_markdown)


if __name__ == "__main__":
    unittest.main()
