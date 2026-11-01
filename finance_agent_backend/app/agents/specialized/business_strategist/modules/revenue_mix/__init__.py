"""Where does revenue come from, in numbers?"""

from app.agents.specialized.business_strategist.modules.revenue_mix.calculate import (
    clean_name,
    extract_segment_disclosures,
    latest_year_figures,
    match_name,
    name_key,
    reportable_segment_names,
    top_level_line_names,
)
from app.agents.specialized.business_strategist.modules.revenue_mix.render import render_revenue_mix_markdown, render_segment_table_markdown
from app.agents.specialized.business_strategist.modules.revenue_mix.schema import RevenueLineFigure, SegmentExtraction, SegmentFigure

__all__ = [
    "RevenueLineFigure",
    "SegmentExtraction",
    "SegmentFigure",
    "clean_name",
    "extract_segment_disclosures",
    "latest_year_figures",
    "match_name",
    "name_key",
    "render_revenue_mix_markdown",
    "render_segment_table_markdown",
    "reportable_segment_names",
    "top_level_line_names",
]
