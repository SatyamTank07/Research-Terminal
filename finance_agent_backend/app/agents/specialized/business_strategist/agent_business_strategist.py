"""Business Strategist: what the company sells, where revenue comes from, what protects it and who it depends on.

`analyze` is the checklist, deterministic first:
1. context: Item 1 (Business) in filing order, located by its label or the filing's own headings, and the whole
   filing (`context.py`).
2. revenue_mix: segment and product-line revenue parsed and reconciled from the 10-K tables.
3. market_power gather: customer-concentration and pricing sentences from the rest of the filing.
4. judge: one tool-free LLM call proposes the qualitative claims, each with verbatim quotes.
5. verify (`build_strategy`): competition_and_moat, market_power and business_model keep only claims whose quotes
   exist in the filing and evidence their mechanism; `assemble.py` builds the output, data quality and citations.

If the LLM fails or returns malformed JSON, the code-built parts are still returned and the qualitative fields are
marked as not assessed.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.llm_call import complete_once
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.business_strategist.assemble import assemble_strategy, not_assessed_reason, record_extraction
from app.agents.specialized.business_strategist.context import (
    FilingChunk,
    Pool,
    filing_items,
    load_filing,
    locate_item1,
    pool_chunk,
    read_item1,
)
from app.agents.specialized.business_strategist.modules.business_model import verify_business_model
from app.agents.specialized.business_strategist.modules.competition_and_moat import (
    not_assessed_competition,
    verify_competition_and_moat,
)
from app.agents.specialized.business_strategist.modules.market_power import (
    DisclosureSentence,
    gather_disclosures,
    verify_market_power,
)
from app.agents.specialized.business_strategist.modules.revenue_mix import (
    SegmentExtraction,
    extract_segment_disclosures,
    render_revenue_mix_markdown,
    render_segment_table_markdown,
    reportable_segment_names,
    top_level_line_names,
)
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput, StrategistDataQuality

logger = logging.getLogger("finance_agent.agents.business_strategist")

JUDGEMENT_MAX_TOKENS = 2_000


@AgentRegistry.register("business_strategist")
class BusinessStrategistAgent(StructuredAgent[BusinessMoatOutput]):
    """Deterministic-first business model, segment and moat analysis with one verified LLM judgement."""

    prompt_name = "business_strategist"
    tools: List[Any] = []
    output_schema = BusinessMoatOutput

    def analyze(self, ticker: str, fiscal_year: int, callbacks: Optional[List[Any]] = None) -> BusinessMoatOutput:
        """Analyzes the 10-K for `ticker` / `fiscal_year` and returns a verified BusinessMoatOutput. Never raises
        on missing data or LLM failure; problems are recorded in `data_quality`."""
        ticker = ticker.upper()
        dq = StrategistDataQuality()

        # 1. Context: Item 1 and the whole filing.
        item1, located_by = locate_item1(ticker, fiscal_year)
        dq.section_located_by = located_by  # type: ignore[assignment]
        filing, load_notes = load_filing(ticker, fiscal_year)
        dq.notes.extend(load_notes)
        # Stored labels are only trusted when Item 1 itself was found under its label.
        items = filing_items(filing, located_by)

        # 2. revenue_mix: where revenue comes from, in numbers.
        extraction = extract_segment_disclosures(filing, fiscal_year, items)
        record_extraction(extraction, dq)

        text = read_item1(ticker, fiscal_year, item1, located_by)
        pool: Pool = dict(text.pool)
        dq.coverage_mode = text.coverage_mode  # type: ignore[assignment]
        dq.item1_chunks = text.item1_chunks
        dq.chars_read = text.chars_read
        dq.notes.extend(text.notes)

        # 3. market_power gather: concentration and pricing sentences outside Item 1, quotable by the judge.
        disclosures = gather_disclosures(filing, items, set(pool))
        by_id = {c.chunk_id: c for c in filing}
        for d in disclosures:
            chunk = by_id.get(d.chunk_id)
            if chunk is not None:
                pool.setdefault(d.chunk_id, pool_chunk(chunk, ticker, fiscal_year, items))
        dq.supplementary_passages = len(disclosures)

        # 4. Judge: one tool-free call proposes the qualitative claims.
        raw: Optional[Dict[str, Any]] = None
        if text.context or disclosures:
            raw = self._judge(ticker, fiscal_year, text.context, dq, extraction, disclosures, callbacks)
        else:
            dq.notes.append("No Item 1 text or supplementary disclosures were available; qualitative analysis skipped.")

        logger.info(
            f"BusinessStrategist {ticker} FY{fiscal_year}: located_by={located_by}, mode={dq.coverage_mode}, "
            f"chars={dq.chars_read}, segments={len({f.segment_name for f in extraction.segments})}, "
            f"revenue_lines={len({f.name for f in extraction.revenue_lines})}, llm={dq.llm_status}"
        )
        # 5. Verify each claim against the filing and assemble.
        return build_strategy(ticker, fiscal_year, raw, pool, extraction, disclosures, filing, items, dq)

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
        prompt = render_prompt(
            "business_strategist", "prompt_business_strategist_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            coverage_mode=dq.coverage_mode,
            section_context=context,
            disclosures=disclosures,
            segment_basis=extraction.segment_basis,
            segment_names=reportable_segment_names(extraction),
            line_names=top_level_line_names(extraction),
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


def build_strategy(
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
    """Verifies the judge's claims module by module and assembles the output; `raw` is None when no judgement ran.

    Moat and threats, then market power, then business model: none reads another's result, and this order keeps
    the data-quality notes and the evidence in the order the report has always shown them.
    """
    data = raw or {}
    assessed = raw is not None
    reason = not_assessed_reason(dq.llm_status)
    competition = verify_competition_and_moat(data, pool, extraction) if assessed else not_assessed_competition(reason)
    market = verify_market_power(data, pool, disclosures, assessed, reason)
    business = verify_business_model(data, pool, extraction, reason)
    return assemble_strategy(ticker, fiscal_year, extraction, competition, market, business, pool, filing, items, dq)


def analyze_business(run: RunContext, upstream: Upstream) -> BusinessMoatOutput:
    """Pipeline entry: Item 1 business model and moat for the run's filing."""
    return BusinessStrategistAgent().analyze(ticker=run.ticker, fiscal_year=run.fiscal_year, callbacks=run.callbacks)


def summarize_moat(moat: Optional[BusinessMoatOutput]) -> StatusLine:
    if moat is None:
        return "Economic moat analysis unavailable", None
    return f"Completed economic moat analysis (Moat: {moat.economic_moat_type})", None
