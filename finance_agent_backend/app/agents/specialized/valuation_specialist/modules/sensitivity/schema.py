"""Result types of the sensitivity module."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SensitivityGrid:
    """Fair value per share (unrounded; None = no finite value) for WACC rows x one value driver's columns."""

    wacc_rows: Tuple[float, ...]
    columns: Tuple[float, ...]
    cells: Tuple[Tuple[Optional[float], ...], ...]
    base_row: int
    base_cell: Optional[Tuple[int, int]]   # the headline value, shown in bold; None if its column was floored away


@dataclass(frozen=True)
class Sensitivity:
    growth_grid: SensitivityGrid
    ronic_grid: Optional[SensitivityGrid]    # value-driver terminal value only
    ronic_at_wacc_rate: Optional[float]      # terminal RONIC when it sits at WACC (the growth grid is then flat)
    sensitivity_matrix_markdown: str
    ronic_sensitivity_matrix_markdown: str
