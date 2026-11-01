"""Synthetic Item 1A chunks shared by the Risk Analyst tests."""

from app.agents.shared.filing.narrative_search import NarrativeChunkResult

CONCENTRATION = (
    "We receive a significant amount of our revenue from a limited number of customers. "
    "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
    "another direct customer represented 14% of total revenue, all of which were primarily attributable "
    "to the Compute & Networking segment."
)


def item1a_chunk(chunk_id: str, content: str, idx: int = 0, item: str = "Item 1A", fy: int = 2026) -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        ticker="TEST",
        fiscal_year=fy,
        item=item,
        breadcrumb=f"TEST > 10-K FY{fy} > PART I > {item}: Risk Factors",
        sub_section="Risk Factors",
        content=content,
        chunk_index=idx,
    )
