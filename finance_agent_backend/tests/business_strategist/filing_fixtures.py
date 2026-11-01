"""Synthetic 10-K chunks for the Strategist tests: an income statement, a segment note and a revenue disaggregation."""

from typing import List, Optional, Sequence

from app.agents.specialized.business_strategist.context import FilingChunk


def md(rows: Sequence[Sequence[str]], header: Sequence[str] = ("", "2025", "2024"), title: Optional[str] = None) -> str:
    lines: List[str] = []
    if title:
        lines += [f"### {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join([":---"] * len(header)) + " |")
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def table(idx: int, content: str, item: str = "Item 8") -> FilingChunk:
    return FilingChunk(chunk_id=f"t{idx}", chunk_index=idx, chunk_type="table", item=item, content=content)


def text(idx: int, content: str, item: str = "Item 8") -> FilingChunk:
    return FilingChunk(chunk_id=f"n{idx}", chunk_index=idx, chunk_type="narrative", item=item, content=content)


INCOME_STATEMENT = md([
    ("Total net sales", "1,000", "850"),
    ("Cost of sales", "600", "520"),
    ("Operating income", "110", "80"),
    ("Net income", "90", "60"),
    ("Diluted earnings per share", "1.00", "0.70"),
])

STACKED_SEGMENTS = md(
    [
        ("Retail", "", ""),
        ("Net sales", "700", "800"),
        ("Operating expenses", "670", "760"),
        ("Operating income", "30", "40"),
        ("Cloud", "", ""),
        ("Net sales", "150", "200"),
        ("Operating expenses", "100", "130"),
        ("Operating income (loss)", "50", "70"),
        ("Consolidated", "", ""),
        ("Net sales", "850", "1,000"),
        ("Operating income", "80", "110"),
    ],
    header=("", "2024", "2025"),  # ascending period columns
)

DISAGGREGATION = md([
    ("Net sales:", "", ""),
    ("Online stores (1)", "500", "450"),
    ("Seller services (2)", "300", "250"),
    ("Cloud services", "200", "150"),
    ("Consolidated", "1,000", "850"),
])


def filing(*extra: FilingChunk, segments: str = STACKED_SEGMENTS, disaggregation: str = DISAGGREGATION) -> List[FilingChunk]:
    chunks = [
        table(1, INCOME_STATEMENT),
        text(2, "Note 10 — Segment Information. We have two reportable segments reviewed by our CODM."),
        table(3, segments),
        text(4, "Net sales by groups of similar products and services is as follows:"),
        table(5, disaggregation),
    ]
    return chunks + list(extra)
