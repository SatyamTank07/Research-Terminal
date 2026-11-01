"""Builds the Strategist's output from the verified module results: the report fields, the data-quality record
(what was read, extracted, proposed and dropped) and the citations with true 10-K items."""

from typing import Any, Dict, List, Sequence

from app.agents.specialized.business_strategist.context import FilingChunk, Pool
from app.agents.specialized.business_strategist.evidence import StrategyEvidence, VerifyTally
from app.agents.specialized.business_strategist.modules.business_model import BusinessModel
from app.agents.specialized.business_strategist.modules.competition_and_moat import CompetitionAndMoat
from app.agents.specialized.business_strategist.modules.market_power import MarketPower
from app.agents.specialized.business_strategist.modules.revenue_mix import (
    SegmentExtraction,
    render_revenue_mix_markdown,
    render_segment_table_markdown,
    reportable_segment_names,
)
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput, StrategistDataQuality


def not_assessed_reason(llm_status: str) -> str:
    """Why the qualitative fields carry no judgement."""
    if llm_status == "failed":
        return "the qualitative LLM call failed"
    if llm_status == "malformed":
        return "the model returned malformed output"
    return "no filing text was available"


def record_extraction(extraction: SegmentExtraction, dq: StrategistDataQuality) -> None:
    dq.segment_tables_found = extraction.segment_tables_found
    dq.segment_layout = extraction.segment_layout
    dq.segment_basis = extraction.segment_basis  # type: ignore[assignment]
    dq.segment_reconciliation = extraction.segment_reconciliation  # type: ignore[assignment]
    dq.revenue_tables_found = extraction.revenue_tables_found
    dq.revenue_lines_reconciliation = extraction.revenue_lines_reconciliation  # type: ignore[assignment]
    dq.consolidated_revenue = extraction.consolidated_revenue
    dq.units = extraction.units
    dq.notes.extend(extraction.notes)


def record_tally(tally: VerifyTally, dq: StrategistDataQuality) -> None:
    dq.claims_proposed += tally.claims_proposed
    dq.claims_dropped += tally.claims_dropped
    dq.quotes_verified += tally.quotes_verified
    dq.quotes_rejected += tally.quotes_rejected
    dq.quotes_off_topic += tally.quotes_off_topic
    dq.notes.extend(tally.notes)


def assemble_strategy(
    ticker: str,
    fiscal_year: int,
    extraction: SegmentExtraction,
    competition: CompetitionAndMoat,
    market: MarketPower,
    business: BusinessModel,
    pool: Pool,
    filing: Sequence[FilingChunk],
    items: Dict[str, str],
    dq: StrategistDataQuality,
) -> BusinessMoatOutput:
    """The Strategist's output. Tallies, notes and evidence follow the verify order: moat and threats, market
    power, then business model."""
    for tally in (competition.tally, market.tally, business.tally):
        record_tally(tally, dq)
    evidence: List[StrategyEvidence] = competition.evidence + market.evidence
    dq.claims_verified = len({(e.claim, e.subject) for e in evidence})
    return BusinessMoatOutput(
        ticker=ticker,
        fiscal_year=fiscal_year,
        business_summary=business.business_summary,
        revenue_architecture=business.revenue_architecture,
        reportable_segments=reportable_segment_names(extraction),
        primary_product_segments=business.primary_product_segments,
        segment_details=business.segment_details,
        segment_financials=extraction.segments,
        revenue_lines=extraction.revenue_lines,
        segment_table_markdown=render_segment_table_markdown(extraction.segments),
        revenue_mix_markdown=render_revenue_mix_markdown(extraction.revenue_lines),
        competitive_threats=competition.competitive_threats,
        pricing_power_assessment=market.pricing_power_assessment,
        customer_concentration=market.customer_concentration,
        evidence=evidence,
        data_quality=dq,
        citations=build_citations(evidence, pool, extraction, filing, items, ticker, fiscal_year),
        **competition.moat_fields(),
    )


def build_citations(
    evidence: Sequence[StrategyEvidence],
    pool: Pool,
    extraction: SegmentExtraction,
    filing: Sequence[FilingChunk],
    items: Dict[str, str],
    ticker: str,
    fiscal_year: int,
) -> List[Dict[str, Any]]:
    """One citation per chunk: the evidence chunks first, then the segment and revenue tables."""
    by_id = {c.chunk_id: c for c in filing}
    citations: List[Dict[str, Any]] = []
    seen = set()

    def add(chunk_id: str, fallback: Dict[str, Any]) -> None:
        if chunk_id in seen:
            return
        seen.add(chunk_id)
        citation: Dict[str, Any] = {"chunk_id": chunk_id, "ticker": ticker, "fiscal_year": fiscal_year}
        for key in ("item", "breadcrumb", "sub_section"):
            if fallback.get(key) is not None:
                citation[key] = fallback[key]
        citations.append(citation)

    for ev in evidence:
        add(ev.chunk_id, pool.get(ev.chunk_id, {"item": ev.item, "breadcrumb": ev.breadcrumb}))
    table_ids = list(extraction.segment_source_chunk_ids)
    if extraction.revenue_lines_source_chunk_id:
        table_ids.append(extraction.revenue_lines_source_chunk_id)
    for cid in table_ids:
        chunk = by_id.get(cid)
        if chunk is not None:
            add(cid, {"item": items.get(cid) or chunk.item, "breadcrumb": chunk.breadcrumb, "sub_section": chunk.sub_section})
    return citations
