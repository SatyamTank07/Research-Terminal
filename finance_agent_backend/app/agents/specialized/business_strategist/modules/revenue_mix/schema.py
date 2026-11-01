"""Segment and product-line revenue figures extracted from the 10-K tables (USD millions)."""

from typing import List, Optional

from pydantic import BaseModel, Field

UNITS_MILLIONS = "USD millions"


class SegmentFigure(BaseModel):
    """Revenue and segment profit for one reportable segment and fiscal year (USD millions)."""

    segment_name: str
    fiscal_year: int
    revenue: float
    segment_profit: Optional[float] = None
    profit_measure: Optional[str] = None
    revenue_share_pct: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    profit_share_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    reportable: bool = True
    source_chunk_id: str


class RevenueLineFigure(BaseModel):
    """Net sales of one disaggregated product / service line for one fiscal year (USD millions)."""

    name: str
    fiscal_year: int
    revenue: float
    revenue_share_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    parent: Optional[str] = None
    source_chunk_id: str


class SegmentExtraction(BaseModel):
    """Result of the deterministic segment / revenue-line extraction for one filing."""

    segments: List[SegmentFigure] = Field(default_factory=list)
    revenue_lines: List[RevenueLineFigure] = Field(default_factory=list)
    segment_layout: Optional[str] = None
    segment_basis: str = "not_found"
    segment_reconciliation: str = "not_applicable"
    segment_source_chunk_ids: List[str] = Field(default_factory=list)
    revenue_lines_source_chunk_id: Optional[str] = None
    revenue_lines_reconciliation: str = "not_applicable"
    consolidated_revenue: Optional[float] = None
    consolidated_revenue_year: Optional[int] = None
    latest_fiscal_year: Optional[int] = None
    units: str = UNITS_MILLIONS
    segment_tables_found: int = 0
    revenue_tables_found: int = 0
    notes: List[str] = Field(default_factory=list)
