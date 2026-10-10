"""Deterministic parsing and rendering of SEC financial statement tables.

SEC iXBRL statements split one logical value across several HTML cells (a lone
``$`` cell, the number, a lone ``)`` cell) and use colspans that leave values
misaligned against the header row. This module reduces every row to its label
plus the ordered numeric tokens it carries, then maps those tokens positionally
onto the fiscal periods detected in the header rows.

Shared by the ingestion parser (clean markdown for new filings) and the
Financial Auditor's statement extractor (re-parsing chunks already stored in
the database, no re-ingestion required).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_YEAR_RE = re.compile(r"(?<![\d,.])((?:19|20)\d{2})(?![\d,.]\d)")
_DASH_TOKENS = {"—", "–", "-", "−", "— —"}
_SKIP_TOKENS = {"", "$", "%", ")", "(", "$)", "(%"}
_FOOTNOTE_RE = re.compile(r"\s*\((?:\d{1,2}|[a-z])\)\s*$")
_MD_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}")
_PLACEHOLDER_HEADER_RE = re.compile(r"^Col_\d+$")


@dataclass
class StatementRow:
    """A single statement line: its label, values by fiscal year, and enclosing section header."""

    label: str
    norm_label: str
    values: Dict[int, float]
    tokens: List[float]
    index: int
    section: Optional[str] = None
    is_header: bool = False
    complete: bool = True


@dataclass
class StatementGrid:
    """A parsed financial statement table with detected fiscal periods."""

    periods: List[int]
    rows: List[StatementRow] = field(default_factory=list)
    title: Optional[str] = None
    scale_divisor: float = 1.0
    chunk_id: Optional[str] = None
    filing_year: Optional[int] = None

    @property
    def value_rows(self) -> List[StatementRow]:
        return [r for r in self.rows if not r.is_header]

    def index_of(self, pattern: str) -> Optional[int]:
        """Returns the row index of the first value row whose normalized label matches the regex."""
        rx = re.compile(pattern)
        for row in self.value_rows:
            if rx.search(row.norm_label):
                return row.index
        return None


def normalize_label(label: str) -> str:
    """Lowercases a line-item label, unifies quotes and whitespace, and strips footnote markers."""
    s = (label or "").replace("’", "'").replace("‘", "'").replace("`", "'")
    s = " ".join(s.split()).strip().lower()
    s = _FOOTNOTE_RE.sub("", s)
    return s.rstrip(":,;").strip()


_NOT_NUMERIC = object()


def parse_value_token(cell: str):
    """
    Parses one table cell into a float.

    Returns None for structural filler (empty, lone ``$``/``%``/``)``), 0.0 for dashes,
    a negative float for parenthesized amounts, and ``_NOT_NUMERIC`` for text.
    """
    s = (cell or "").replace(" ", " ").replace(" ", " ").strip()
    if s in _SKIP_TOKENS:
        return None
    if s in _DASH_TOKENS:
        return 0.0
    negative = s.startswith("(") or s.endswith(")") or s.startswith("-") or s.startswith("−")
    core = re.sub(r"[\s$%,()−\-]", "", s)
    if not core:
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?", core):
        return _NOT_NUMERIC
    value = float(core)
    return -value if negative else value


def _detect_scale_divisor(text: str) -> float:
    """Converts '(in thousands)' / '(in billions)' captions into a divisor that yields $ millions."""
    lower = text.lower()
    if "in thousands" in lower:
        return 1000.0
    if "in billions" in lower:
        return 0.001
    return 1.0


def parse_statement_rows(
    rows: List[List[str]],
    title: Optional[str] = None,
    context_text: str = "",
) -> StatementGrid:
    """
    Parses raw cell rows (first cell = label) into a StatementGrid.

    Periods are the distinct years found in the leading rows before the first row that
    carries numeric values. Value tokens are mapped positionally onto those periods.
    """
    periods: List[int] = []
    body_start = 0
    for i, row in enumerate(rows):
        cells = [c for c in row if c is not None]
        numeric = [
            t for t in (parse_value_token(c) for c in cells[1:])
            if t is not None and t is not _NOT_NUMERIC
        ]
        row_years = [int(y) for c in cells for y in _YEAR_RE.findall(c)]
        bare_year_cells = all(
            _YEAR_RE.fullmatch(c.strip()) for c in cells[1:] if parse_value_token(c) not in (None, _NOT_NUMERIC)
        )
        if numeric and not (row_years and bare_year_cells):
            body_start = i
            break
        if periods and not row_years:
            # First label-only row after the period header (e.g. "Revenues") starts the body.
            body_start = i
            break
        for y in row_years:
            if y not in periods:
                periods.append(y)
        body_start = i + 1

    grid = StatementGrid(
        periods=periods,
        title=title,
        scale_divisor=_detect_scale_divisor(f"{title or ''} {context_text}"),
    )

    section: Optional[str] = None
    for i, row in enumerate(rows[body_start:], start=body_start):
        if not row:
            continue
        label = (row[0] or "").strip().strip("*").strip()
        tokens: List[float] = []
        for cell in row[1:]:
            tok = parse_value_token(cell)
            if tok is None or tok is _NOT_NUMERIC:
                continue
            tokens.append(tok)

        if not label and not tokens:
            continue
        if not tokens:
            section = label
            grid.rows.append(
                StatementRow(label=label, norm_label=normalize_label(label), values={},
                             tokens=[], index=i, section=None, is_header=True)
            )
            continue

        values: Dict[int, float] = {}
        for period, tok in zip(periods, tokens):
            values[period] = tok
        grid.rows.append(
            StatementRow(
                label=label,
                norm_label=normalize_label(label),
                values=values,
                tokens=tokens,
                index=i,
                section=normalize_label(section) if section else None,
                complete=len(tokens) >= len(periods) > 0,
            )
        )
    return grid


def markdown_to_rows(markdown: str):
    """Splits a stored markdown table chunk into (title, cell rows), dropping placeholder headers."""
    title = None
    rows: List[List[str]] = []
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or title
            continue
        if stripped.startswith("**") and stripped.endswith("**") and not rows:
            title = stripped.strip("*").strip() or title
            continue
        if not stripped.startswith("|"):
            continue
        if _MD_SEPARATOR_RE.match(stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        cells = ["" if _PLACEHOLDER_HEADER_RE.match(c) else c for c in cells]
        rows.append(cells)
    return title, rows


def parse_statement_table(
    markdown: str,
    chunk_id: Optional[str] = None,
    filing_year: Optional[int] = None,
) -> StatementGrid:
    """Parses a stored markdown statement table (including legacy Col_n layouts) into a StatementGrid."""
    title, rows = markdown_to_rows(markdown)
    grid = parse_statement_rows(rows, title=title, context_text=markdown[:400] if markdown else "")
    grid.chunk_id = chunk_id
    grid.filing_year = filing_year
    return grid


def _format_number(value: float) -> str:
    if value == 0:
        return "—"
    magnitude = abs(value)
    body = f"{magnitude:,.2f}" if magnitude != int(magnitude) else f"{magnitude:,.0f}"
    return f"({body})" if value < 0 else body


def is_well_formed(grid: StatementGrid, min_ratio: float = 0.8) -> bool:
    """True when periods were detected and most value rows map cleanly onto them."""
    value_rows = grid.value_rows
    if not grid.periods or not value_rows:
        return False
    complete = sum(1 for r in value_rows if r.complete and len(r.tokens) == len(grid.periods))
    return complete / len(value_rows) >= min_ratio


def render_clean_statement(grid: StatementGrid, period_label: str = "FY") -> str:
    """Renders a StatementGrid as a compact `| Line item | FY2025 | FY2024 |` markdown table."""
    if not grid.periods:
        return ""
    lines = []
    if grid.title:
        lines.append(f"**{grid.title}**\n")
    header = ["Line item"] + [f"{period_label}{p}" for p in grid.periods]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| :--- | " + " | ".join(["---:"] * len(grid.periods)) + " |")
    for row in grid.rows:
        if row.is_header:
            lines.append(f"| **{row.label}** | " + " | ".join([""] * len(grid.periods)) + " |")
            continue
        cells = [_format_number(row.values[p]) if p in row.values else "" for p in grid.periods]
        lines.append(f"| {row.label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
