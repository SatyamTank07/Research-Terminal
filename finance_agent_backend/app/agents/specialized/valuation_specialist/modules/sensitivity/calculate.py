"""How much does the value move if WACC, g or RONIC is off? WACC x terminal growth and WACC x terminal RONIC
grids of fair value per share, re-valued from the base case's validated DCF inputs."""

from dataclasses import replace
from typing import List, Optional

from app.agents.specialized.valuation_specialist.modules.intrinsic_value.calculate import value_point
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.schema import DCFCase, DCFValue
from app.agents.specialized.valuation_specialist.modules.sensitivity.render import render_growth_grid, render_ronic_grid
from app.agents.specialized.valuation_specialist.modules.sensitivity.schema import Sensitivity, SensitivityGrid
from app.agents.specialized.valuation_specialist.modules.sensitivity.thresholds import (
    RONIC_EQUALS_WACC_TOLERANCE,
    RONIC_GRID_MIN_SPREAD_OVER_G,
    SENSITIVITY_GROWTH_STEPS,
    SENSITIVITY_RONIC_STEPS,
    SENSITIVITY_WACC_STEPS,
)


def _per_share(case: DCFCase, rate: float, g: float) -> Optional[float]:
    point = value_point(case, rate, g)
    return None if point is None else point.per_share


def build_sensitivity_grids(dcf: DCFValue) -> Sensitivity:
    """WACC x terminal growth and (value-driver TV only) WACC x terminal RONIC grids around the base case."""
    case, base = dcf.case, dcf.point
    base_row = SENSITIVITY_WACC_STEPS.index(0.0)
    wacc_rows = tuple(round(case.wacc + d, 4) for d in SENSITIVITY_WACC_STEPS)
    growth_cols = tuple(round(case.terminal_growth_rate + d, 4) for d in SENSITIVITY_GROWTH_STEPS)
    growth_base = (base_row, SENSITIVITY_GROWTH_STEPS.index(0.0))
    growth_grid = SensitivityGrid(
        wacc_rows,
        growth_cols,
        tuple(
            tuple(base.per_share if (i, j) == growth_base else _per_share(case, w, g) for j, g in enumerate(growth_cols))
            for i, w in enumerate(wacc_rows)
        ),
        base_row,
        growth_base,
    )

    # WACC vs. terminal RONIC: informative even when RONIC = WACC flattens the growth grid. Columns are base
    # RONIC +/- steps, floored at g + 1 pp so every cell converges.
    ronic_grid = None
    if case.terminal_method == "value_driver":
        floor = case.terminal_growth_rate + RONIC_GRID_MIN_SPREAD_OVER_G
        ronic_cols: List[float] = []
        for d in SENSITIVITY_RONIC_STEPS:
            col = round(max(case.ronic + d, floor), 4)
            if col not in ronic_cols:
                ronic_cols.append(col)
        base_col = round(case.ronic, 4)
        base_cell = (base_row, ronic_cols.index(base_col)) if base_col in ronic_cols else None
        ronic_grid = SensitivityGrid(
            wacc_rows,
            tuple(ronic_cols),
            tuple(
                tuple(
                    base.per_share if (i, k) == base_cell
                    else _per_share(replace(case, ronic=r), w, case.terminal_growth_rate)
                    for k, r in enumerate(ronic_cols)
                )
                for i, w in enumerate(wacc_rows)
            ),
            base_row,
            base_cell,
        )

    at_wacc = case.terminal_method == "value_driver" and abs(case.ronic - case.wacc) <= RONIC_EQUALS_WACC_TOLERANCE
    ronic_at_wacc_rate = case.ronic if at_wacc else None
    return Sensitivity(
        growth_grid=growth_grid,
        ronic_grid=ronic_grid,
        ronic_at_wacc_rate=ronic_at_wacc_rate,
        sensitivity_matrix_markdown=render_growth_grid(growth_grid, ronic_at_wacc_rate),
        ronic_sensitivity_matrix_markdown=render_ronic_grid(ronic_grid) if ronic_grid is not None else "",
    )
