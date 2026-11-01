"""Checks the judge's business description against the code-extracted revenue lines and segments.

segment_details and primary_product_segments follow the extracted revenue lines, else the (non-geographic)
reportable segments, and the judge's names must match them. Without either, a judge-proposed product line is
kept only if the filing text names it. Missing prose falls back to text built from the extracted tables.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.agents.shared.filing.sentences import normalize_text
from app.agents.specialized.business_strategist.context import Pool
from app.agents.specialized.business_strategist.evidence import VerifyTally, not_assessed, text_of
from app.agents.specialized.business_strategist.modules.business_model.schema import SegmentDetail
from app.agents.specialized.business_strategist.modules.business_model.thresholds import MAX_GROWTH_DRIVERS
from app.agents.specialized.business_strategist.modules.revenue_mix import (
    SegmentExtraction,
    clean_name,
    latest_year_figures,
    match_name,
    name_key,
    reportable_segment_names,
    top_level_line_names,
)


@dataclass(frozen=True)
class BusinessModel:
    """The business description and product lines reconciled to the extracted tables, and the dropped names."""

    business_summary: str
    revenue_architecture: str
    segment_details: List[SegmentDetail]
    primary_product_segments: List[str]
    tally: VerifyTally


def verify_business_model(data: Dict[str, Any], pool: Pool, extraction: SegmentExtraction, reason: str) -> BusinessModel:
    """The judge's summary, revenue model and product lines, reconciled to the extracted tables."""
    tally = VerifyTally()
    details, products = reconcile_product_lines(data, pool, extraction, tally)
    return BusinessModel(
        business_summary=text_of(data.get("business_summary")) or not_assessed(reason),
        revenue_architecture=text_of(data.get("revenue_architecture")) or deterministic_architecture(extraction, reason),
        segment_details=details,
        primary_product_segments=products,
        tally=tally,
    )


def reconcile_product_lines(
    data: Dict[str, Any], pool: Pool, extraction: SegmentExtraction, tally: VerifyTally
) -> Tuple[List[SegmentDetail], List[str]]:
    """segment_details / primary_product_segments follow the code-extracted revenue lines, else the
    (non-geographic) reportable segments.

    Without either, a model-proposed product name is kept only if it appears in the filing text.
    """
    line_names = top_level_line_names(extraction)
    segment_names = reportable_segment_names(extraction)
    raw_details = data.get("segment_details")
    proposed: List[Dict[str, Any]] = [d for d in raw_details if isinstance(d, dict)] if isinstance(raw_details, list) else []
    targets = line_names or (segment_names if extraction.segment_basis != "geographic" else [])
    if targets:
        lookup = {name_key(n): n for n in targets}
        by_target: Dict[str, Dict[str, Any]] = {}
        unmatched = []
        for d in proposed:
            canonical = match_name(str(d.get("name") or ""), lookup)
            if canonical is None:
                unmatched.append(str(d.get("name") or ""))
                continue
            by_target.setdefault(canonical, d)
        if unmatched:
            tally.notes.append(
                f"{len(unmatched)} model-proposed product line(s) not in the extracted revenue table were dropped "
                f"({', '.join(n for n in unmatched[:3] if n)})."
            )
        figures = {name_key(f.name): f for f in latest_year_figures(extraction.revenue_lines)}
        figures.update({name_key(f.segment_name): f for f in latest_year_figures(extraction.segments)})
        details = []
        for name in targets:
            d = by_target.get(name, {})
            details.append(
                SegmentDetail(
                    name=name,
                    description=text_of(d.get("description")) or figure_description(name, figures.get(name_key(name))),
                    growth_drivers=growth_drivers(d.get("growth_drivers")),
                )
            )
        return details, list(targets)

    corpus = normalize_text(" ".join(c.get("content", "") for c in pool.values()))
    details = []
    for d in proposed:
        name = clean_name(str(d.get("name") or ""))
        if not name or normalize_text(name) not in corpus:
            if name:
                tally.notes.append(f"Product line '{name}' dropped: not named in the filing text.")
            continue
        details.append(
            SegmentDetail(name=name, description=text_of(d.get("description")) or name, growth_drivers=growth_drivers(d.get("growth_drivers")))
        )
    return details, [d.name for d in details]


def growth_drivers(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()][:MAX_GROWTH_DRIVERS]


def figure_description(name: str, figure: Any) -> str:
    """A product line the judge skipped, described from its extracted revenue figure."""
    if figure is None:
        return f"{name} (reported revenue line)."
    share = f", {figure.revenue_share_pct:.1f}% of revenue" if figure.revenue_share_pct is not None else ""
    return f"{name}: FY{figure.fiscal_year} revenue ${figure.revenue:,.0f}M{share} (from the 10-K tables)."


def deterministic_architecture(extraction: SegmentExtraction, reason: str) -> str:
    """The revenue model as the extracted product-line mix, when the judge gave none."""
    lines = [f for f in latest_year_figures(extraction.revenue_lines) if f.parent is None]
    if not lines:
        return not_assessed(reason)
    mix = ", ".join(
        f"{f.name} {f.revenue_share_pct:.1f}%" for f in lines if f.revenue_share_pct is not None
    )
    return f"Revenue mix by product / service line (FY{lines[0].fiscal_year}, from the 10-K revenue disaggregation): {mix}."
