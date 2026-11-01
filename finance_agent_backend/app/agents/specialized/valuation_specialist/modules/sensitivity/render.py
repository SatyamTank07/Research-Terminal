"""The sensitivity tables: format the unrounded per-share values; the base cell is the headline in bold."""

from typing import List, Optional

from app.agents.specialized.valuation_specialist.modules.sensitivity.schema import SensitivityGrid


def _cell(value: Optional[float], bold: bool) -> str:
    if bold:
        return f"**${value:.2f}**"
    return "N/A" if value is None else f"${value:.2f}"


def _rows(grid: SensitivityGrid) -> List[str]:
    rows = []
    for i, w in enumerate(grid.wacc_rows):
        cells = [f"**{w * 100:.1f}%**" + (" *(Base)*" if i == grid.base_row else "")]
        cells += [_cell(v, (i, j) == grid.base_cell) for j, v in enumerate(grid.cells[i])]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def render_growth_grid(grid: SensitivityGrid, ronic_at_wacc_rate: Optional[float]) -> str:
    lines = [
        "| WACC \\ Growth | " + " | ".join(f"{g * 100:.1f}%" for g in grid.columns) + " |",
        "|:---|" + "|".join([":---:"] * len(grid.columns)) + "|",
        *_rows(grid),
    ]
    if ronic_at_wacc_rate is not None:
        lines.append("")
        lines.append(
            f"_Terminal RONIC ({ronic_at_wacc_rate * 100:.1f}%) is at WACC, so growth beyond the horizon creates no value and the "
            f"growth columns barely move; the WACC x RONIC grid shows what returns on new capital are worth._"
        )
    return "\n".join(lines)


def render_ronic_grid(grid: SensitivityGrid) -> str:
    return "\n".join([
        "| WACC \\ RONIC | " + " | ".join(f"{r * 100:.1f}%" for r in grid.columns) + " |",
        "|:---|" + "|".join([":---:"] * len(grid.columns)) + "|",
        *_rows(grid),
    ])
