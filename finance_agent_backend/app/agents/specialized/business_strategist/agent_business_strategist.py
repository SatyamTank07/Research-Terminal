"""Business & Moat Strategist Agent (deterministic-first, evidence-first).

Code reads the filing before the LLM runs:
- Item 1 (Business) in filing order, located by its section label or, when the stored labels are
  wrong, by the filing's own headings (`locate_section_chunks`).
- The segment note (revenue and the segment profit measure by reportable segment, multi-year) and
  the revenue-disaggregation table (net sales by product / service line), parsed and reconciled
  in Python (`business_disclosure_tools`).
- Customer-concentration and pricing sentences from the rest of the filing.

One tool-free LLM call then supplies the qualitative judgement (business summary, moat
classification with verbatim quotes, competitive threats, pricing power, customer concentration).
Code verifies every quote against the pre-fetched chunks, drops unsupported claims, downgrades an
unsupported moat, reconciles product lines against the disaggregation table and rebuilds citations
with true 10-K items. If the LLM fails or returns malformed JSON the deterministic parts are still
returned, with the qualitative fields marked as not assessed.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.llm_call import complete_once
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.business_strategist.context import (
    FilingChunk,
    Pool,
    filing_items,
    load_filing,
    locate_item1,
    pool_chunk,
    read_item1,
)
from app.agents.specialized.business_strategist.evidence import StrategyEvidence, VerifyTally
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput, StrategistDataQuality
from app.agents.specialized.business_strategist.modules.business_model import verify_business_model
from app.agents.specialized.business_strategist.modules.competition_and_moat import (
    not_assessed_competition,
    verify_competition_and_moat,
)
from app.agents.specialized.business_strategist.modules.market_power import DisclosureSentence, gather_disclosures, verify_market_power
from app.agents.specialized.business_strategist.modules.revenue_mix import (
    SegmentExtraction,
    extract_segment_disclosures,
    render_revenue_mix_markdown,
    render_segment_table_markdown,
    reportable_segment_names,
    top_level_line_names,
)

logger = logging.getLogger("finance_agent.agents.business_strategist")

JUDGEMENT_MAX_TOKENS = 2_000


@AgentRegistry.register("business_strategist")
class BusinessStrategistAgent(StructuredAgent[BusinessMoatOutput]):
    """Deterministic-first business model, segment and moat analysis with one verified LLM judgement."""

    prompt_name = "business_strategist"
    tools: List[Any] = []
    output_schema = BusinessMoatOutput

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze(self, ticker: str, fiscal_year: int, callbacks: Optional[List[Any]] = None) -> BusinessMoatOutput:
        """Analyzes the 10-K for `ticker` / `fiscal_year` and returns a verified BusinessMoatOutput. Never raises
        on missing data or LLM failure; problems are recorded in `data_quality`."""
        ticker = ticker.upper()
        dq = StrategistDataQuality()

        item1, located_by = locate_item1(ticker, fiscal_year)
        dq.section_located_by = located_by  # type: ignore[assignment]
        filing, load_notes = load_filing(ticker, fiscal_year)
        dq.notes.extend(load_notes)
        # Stored labels are only trusted when Item 1 itself was found under its label.
        items = filing_items(filing, located_by)

        extraction = extract_segment_disclosures(filing, fiscal_year, items)
        self._record_extraction(extraction, dq)

        text = read_item1(ticker, fiscal_year, item1, located_by)
        pool: Pool = dict(text.pool)
        context = text.context
        dq.coverage_mode = text.coverage_mode  # type: ignore[assignment]
        dq.item1_chunks = text.item1_chunks
        dq.chars_read = text.chars_read
        dq.notes.extend(text.notes)

        disclosures = gather_disclosures(filing, items, set(pool))
        by_id = {c.chunk_id: c for c in filing}
        for d in disclosures:
            chunk = by_id.get(d.chunk_id)
            if chunk is not None:
                pool.setdefault(d.chunk_id, pool_chunk(chunk, ticker, fiscal_year, items))
        dq.supplementary_passages = len(disclosures)

        raw: Optional[Dict[str, Any]] = None
        if context or disclosures:
            raw = self._judge(ticker, fiscal_year, context, dq, extraction, disclosures, callbacks)
        else:
            dq.notes.append("No Item 1 text or supplementary disclosures were available; qualitative analysis skipped.")

        logger.info(
            f"BusinessStrategist {ticker} FY{fiscal_year}: located_by={located_by}, mode={dq.coverage_mode}, "
            f"chars={dq.chars_read}, segments={len({f.segment_name for f in extraction.segments})}, "
            f"revenue_lines={len({f.name for f in extraction.revenue_lines})}, llm={dq.llm_status}"
        )
        return self._assemble(ticker, fiscal_year, raw, pool, extraction, disclosures, filing, items, dq)

    # ------------------------------------------------------------------
    # LLM judgement
    # ------------------------------------------------------------------
    def _judge(
        self,
        ticker: str,
        fiscal_year: int,
        context: str,
        dq: StrategistDataQuality,
        extraction: SegmentExtraction,
        disclosures: Sequence[DisclosureSentence],
        callbacks: Optional[List[Any]],
    ) -> Optional[Dict[str, Any]]:
        segment_names = reportable_segment_names(extraction)
        line_names = top_level_line_names(extraction)
        prompt = render_prompt(
            "business_strategist", "prompt_business_strategist_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            coverage_mode=dq.coverage_mode,
            section_context=context,
            disclosures=disclosures,
            segment_basis=extraction.segment_basis,
            segment_names=segment_names,
            line_names=line_names,
            segment_table=render_segment_table_markdown(extraction.segments),
            revenue_mix=render_revenue_mix_markdown(extraction.revenue_lines),
        )
        try:
            content = self._run_judgement_llm(prompt, callbacks)
        except Exception as e:
            logger.warning(f"Business strategist LLM call failed for {ticker} FY{fiscal_year}: {e}")
            dq.llm_status = "failed"
            dq.notes.append(f"Qualitative LLM call failed ({type(e).__name__}); moat and competitive fields not assessed.")
            return None
        parsed = self.parse_json_content(content)
        if not parsed:
            dq.llm_status = "malformed"
            dq.notes.append("The model returned no parsable JSON; moat and competitive fields not assessed.")
            return None
        dq.llm_status = "ok"
        return parsed

    def _run_judgement_llm(self, prompt: str, callbacks: Optional[List[Any]]) -> str:
        """Single tool-free completion returning the qualitative judgement as JSON text."""
        return complete_once(
            self.model_name, prompt, system_prompt=self.get_system_prompt(),
            max_tokens=JUDGEMENT_MAX_TOKENS, max_retries=3, json_mode=True, callbacks=callbacks,
        )

    # ------------------------------------------------------------------
    # Data-quality record
    # ------------------------------------------------------------------
    @staticmethod
    def _record_extraction(extraction: SegmentExtraction, dq: StrategistDataQuality) -> None:
        dq.segment_tables_found = extraction.segment_tables_found
        dq.segment_layout = extraction.segment_layout
        dq.segment_basis = extraction.segment_basis  # type: ignore[assignment]
        dq.segment_reconciliation = extraction.segment_reconciliation  # type: ignore[assignment]
        dq.revenue_tables_found = extraction.revenue_tables_found
        dq.revenue_lines_reconciliation = extraction.revenue_lines_reconciliation  # type: ignore[assignment]
        dq.consolidated_revenue = extraction.consolidated_revenue
        dq.units = extraction.units
        dq.notes.extend(extraction.notes)

    # ------------------------------------------------------------------
    # Assembly & enforcement
    # ------------------------------------------------------------------
    def _assemble(
        self,
        ticker: str,
        fiscal_year: int,
        raw: Optional[Dict[str, Any]],
        pool: Pool,
        extraction: SegmentExtraction,
        disclosures: Sequence[DisclosureSentence],
        filing: Sequence[FilingChunk],
        items: Dict[str, str],
        dq: StrategistDataQuality,
    ) -> BusinessMoatOutput:
        segment_names = reportable_segment_names(extraction)
        data = raw or {}
        assessed = raw is not None
        reason = (
            "the qualitative LLM call failed" if dq.llm_status == "failed"
            else "the model returned malformed output" if dq.llm_status == "malformed"
            else "no filing text was available"
        )

        competition = verify_competition_and_moat(data, pool, extraction) if assessed else not_assessed_competition(reason)
        market = verify_market_power(data, pool, disclosures, assessed, reason)
        evidence: List[StrategyEvidence] = competition.evidence + market.evidence
        business = verify_business_model(data, pool, extraction, reason)
        self._record_tally(competition.tally, dq)
        self._record_tally(market.tally, dq)
        self._record_tally(business.tally, dq)

        dq.claims_verified = len({(e.claim, e.subject) for e in evidence})
        return BusinessMoatOutput(
            ticker=ticker,
            fiscal_year=fiscal_year,
            business_summary=business.business_summary,
            revenue_architecture=business.revenue_architecture,
            reportable_segments=segment_names,
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
            citations=self._build_citations(evidence, pool, extraction, filing, items, ticker, fiscal_year),
            **competition.moat_fields(),
        )

    # --- judgement fields -------------------------------------------------
    @staticmethod
    def _record_tally(tally: VerifyTally, dq: StrategistDataQuality) -> None:
        dq.claims_proposed += tally.claims_proposed
        dq.claims_dropped += tally.claims_dropped
        dq.quotes_verified += tally.quotes_verified
        dq.quotes_rejected += tally.quotes_rejected
        dq.quotes_off_topic += tally.quotes_off_topic
        dq.notes.extend(tally.notes)

    # --- citations --------------------------------------------------------
    @staticmethod
    def _build_citations(
        evidence: Sequence[StrategyEvidence],
        pool: Pool,
        extraction: SegmentExtraction,
        filing: Sequence[FilingChunk],
        items: Dict[str, str],
        ticker: str,
        fiscal_year: int,
    ) -> List[Dict[str, Any]]:
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


def analyze_business(run: RunContext, upstream: Upstream) -> BusinessMoatOutput:
    """Pipeline entry: Item 1 business model and moat for the run's filing."""
    return BusinessStrategistAgent().analyze(ticker=run.ticker, fiscal_year=run.fiscal_year, callbacks=run.callbacks)


def summarize_moat(moat: Optional[BusinessMoatOutput]) -> StatusLine:
    if moat is None:
        return "Economic moat analysis unavailable", None
    return f"Completed economic moat analysis (Moat: {moat.economic_moat_type})", None
