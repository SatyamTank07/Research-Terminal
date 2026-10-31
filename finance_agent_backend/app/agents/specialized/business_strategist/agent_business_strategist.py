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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.filing.narrative_search import NarrativeChunkResult, retrieve_10k_narrative
from app.agents.shared.filing.sections import build_section_context, locate_section_chunks
from app.agents.shared.filing.sentences import normalize_text
from app.agents.specialized.business_strategist.state_business_strategist import (
    MOAT_TYPES,
    NOT_DISCLOSED,
    BusinessMoatOutput,
    SegmentDetail,
    SegmentMoat,
    StrategistDataQuality,
    StrategyEvidence,
)
from app.agents.specialized.prompts import render_prompt
from app.agents.tools.business_disclosure_tools import (
    DisclosureSentence,
    FilingChunk,
    SegmentExtraction,
    clean_name,
    extract_segment_disclosures,
    find_disclosure_sentences,
    infer_section_items,
    load_filing_chunks,
    name_key,
    render_revenue_mix_markdown,
    render_segment_table_markdown,
)
from app.agents.tools.claim_relevance import (
    moat_quote_supports,
    pricing_claim_supported,
    segment_quote_mentions,
    threat_quote_supports,
)
from app.agents.tools.risk_disclosure_tools import numbers_supported, verify_quote

logger = logging.getLogger("finance_agent.agents.business_strategist")

ITEM1_SECTION = "Item 1"
ITEM1_CHAR_BUDGET = 90_000
JUDGEMENT_MAX_TOKENS = 2_000
MAX_QUOTE_CHARS = 300
MAX_EVIDENCE_PER_CLAIM = 2
MAX_SECONDARY_MOATS = 2
MAX_THREATS = 4
WIDE_MIN_QUOTES = 2
RAG_FALLBACK_LIMIT = 6
RAG_FALLBACK_QUERIES = (
    "business overview products services customers how the company generates revenue",
    "competition competitive factors competitors market position",
    "intellectual property brand network scale technology advantages",
)

_NOT_DISCLOSED_HINTS = ("not explicitly disclosed", "not disclosed", "no disclosure", "not discussed", "not stated")
_DURABILITY = {"wide": "Wide", "narrow": "Narrow", "none": "None", "indeterminate": "Indeterminate"}
_TRAJECTORY = {"expanding": "Expanding", "stable": "Stable", "deteriorating": "Deteriorating", "indeterminate": "Indeterminate"}
_MOAT_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("network", "Network Effects"),
    ("switching", "High Switching Costs"),
    ("cost advantage", "Cost Advantage"),
    ("low cost", "Cost Advantage"),
    ("intangible", "Intangible Assets / Brand"),
    ("brand", "Intangible Assets / Brand"),
    ("patent", "Intangible Assets / Brand"),
    ("efficient scale", "Efficient Scale"),
    ("no moat", "None"),
)

Pool = Dict[str, Dict[str, Any]]


def _not_assessed(reason: str) -> str:
    return f"Not assessed: {reason}."


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

        item1, located_by = self._safe_locate(ticker, fiscal_year)
        dq.section_located_by = located_by  # type: ignore[assignment]
        filing = self._safe_load_filing(ticker, fiscal_year, dq)
        # Stored labels are only trusted when Item 1 itself was found under its label.
        items = infer_section_items(filing) if located_by == "heading_boundary" else {}

        extraction = extract_segment_disclosures(filing, fiscal_year, items)
        self._record_extraction(extraction, dq)

        if not item1:
            item1 = self._rag_fallback(ticker, fiscal_year)
        pool: Pool = {c.chunk_id: c.model_dump() for c in item1}
        if item1:
            context, mode, chars = build_section_context(item1, ITEM1_CHAR_BUDGET)
            dq.coverage_mode = "rag_fallback" if located_by == "not_found" else mode  # type: ignore[assignment]
        else:
            context, chars = "", 0
            dq.coverage_mode = "no_context"
        dq.item1_chunks = len(item1) if located_by != "not_found" else 0
        dq.chars_read = chars
        self._note_coverage(dq, ticker, fiscal_year)

        disclosures = self._safe_disclosures(filing, items, set(pool))
        by_id = {c.chunk_id: c for c in filing}
        for d in disclosures:
            chunk = by_id.get(d.chunk_id)
            if chunk is not None:
                pool.setdefault(d.chunk_id, self._chunk_dict(chunk, ticker, fiscal_year, items))
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
        segment_names = self._reportable_names(extraction)
        line_names = self._line_names(extraction)
        prompt = render_prompt(
            "prompt_business_strategist_query.j2",
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
        from dotenv import load_dotenv
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        load_dotenv(override=True)
        llm = ChatOpenAI(
            model=self.model_name.replace("openai:", ""),
            temperature=0,
            max_retries=3,
            max_tokens=JUDGEMENT_MAX_TOKENS,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        config = {"callbacks": callbacks} if callbacks else {}
        messages = [SystemMessage(content=self.get_system_prompt()), HumanMessage(content=prompt)]
        return llm.invoke(messages, config=config).content

    # ------------------------------------------------------------------
    # Pre-fetch helpers (never raise)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_locate(ticker: str, fiscal_year: int) -> Tuple[List[NarrativeChunkResult], str]:
        try:
            return locate_section_chunks(ticker, fiscal_year, ITEM1_SECTION)
        except Exception as e:
            logger.warning(f"Item 1 pre-fetch failed for {ticker} FY{fiscal_year}: {e}")
            return [], "not_found"

    @staticmethod
    def _safe_load_filing(ticker: str, fiscal_year: int, dq: StrategistDataQuality) -> List[FilingChunk]:
        try:
            return load_filing_chunks(ticker, fiscal_year)
        except Exception as e:
            logger.warning(f"Filing load failed for {ticker} FY{fiscal_year}: {e}")
            dq.notes.append("Filing tables could not be loaded; segment and revenue-line figures not extracted.")
            return []

    @staticmethod
    def _safe_disclosures(
        filing: Sequence[FilingChunk], items: Dict[str, str], exclude: set
    ) -> List[DisclosureSentence]:
        try:
            return find_disclosure_sentences(filing, items, exclude_chunk_ids=exclude)
        except Exception as e:
            logger.warning(f"Disclosure sentence screen failed: {e}")
            return []

    @staticmethod
    def _rag_fallback(ticker: str, fiscal_year: int) -> List[NarrativeChunkResult]:
        """Hybrid retrieval across the filing when Item 1 has neither a label nor a heading."""
        found: Dict[str, NarrativeChunkResult] = {}
        for query in RAG_FALLBACK_QUERIES:
            try:
                for chunk in retrieve_10k_narrative(ticker, fiscal_year, query, section_item=None, limit=RAG_FALLBACK_LIMIT):
                    found.setdefault(chunk.chunk_id, chunk)
            except Exception as e:
                logger.warning(f"Item 1 retrieval fallback query failed for {ticker} FY{fiscal_year}: {e}")
        return sorted(found.values(), key=lambda c: c.chunk_index)

    @staticmethod
    def _chunk_dict(chunk: FilingChunk, ticker: str, fiscal_year: int, items: Dict[str, str]) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "item": items.get(chunk.chunk_id) or chunk.item,
            "breadcrumb": chunk.breadcrumb,
            "sub_section": chunk.sub_section,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
        }

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

    @staticmethod
    def _note_coverage(dq: StrategistDataQuality, ticker: str, fiscal_year: int) -> None:
        if dq.section_located_by == "heading_boundary":
            dq.notes.append(
                "Item 1 was located from the filing's own section headings because the stored section labels were wrong; "
                "citations use heading-inferred items."
            )
        if dq.coverage_mode == "rag_fallback":
            dq.notes.append(
                f"Item 1 could not be located in the {ticker} FY{fiscal_year} 10-K; business context was retrieved by "
                "search across the filing and may be incomplete."
            )
        elif dq.coverage_mode == "budgeted_section":
            dq.notes.append("Item 1 exceeded the context budget; later paragraphs were truncated to their lead text.")
        elif dq.coverage_mode == "no_context":
            dq.notes.append(f"No Item 1 text could be retrieved for {ticker} FY{fiscal_year}.")

    # ------------------------------------------------------------------
    # Deterministic name lists
    # ------------------------------------------------------------------
    @staticmethod
    def _latest(figures: Sequence[Any]) -> List[Any]:
        if not figures:
            return []
        year = max(f.fiscal_year for f in figures)
        return [f for f in figures if f.fiscal_year == year]

    def _reportable_names(self, extraction: SegmentExtraction) -> List[str]:
        return [f.segment_name for f in self._latest(extraction.segments) if f.reportable]

    def _line_names(self, extraction: SegmentExtraction) -> List[str]:
        return [f.name for f in self._latest(extraction.revenue_lines) if f.parent is None]

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
        segment_names = self._reportable_names(extraction)
        line_names = self._line_names(extraction)
        evidence: List[StrategyEvidence] = []
        data = raw or {}
        assessed = raw is not None
        reason = (
            "the qualitative LLM call failed" if dq.llm_status == "failed"
            else "the model returned malformed output" if dq.llm_status == "malformed"
            else "no filing text was available"
        )

        moat = self._enforce_moats(data, pool, segment_names, line_names, evidence, dq) if assessed else None
        threats = self._enforce_threats(data, pool, evidence, dq) if assessed else []
        pricing = self._enforce_pricing(data, pool, evidence, dq) if assessed else _not_assessed(reason)
        concentration = self._enforce_concentration(data, pool, disclosures, evidence, dq, assessed, reason)
        details, products = self._reconcile_details(data, pool, extraction, line_names, segment_names, dq)

        business_summary = self._text(data.get("business_summary")) or _not_assessed(reason)
        revenue_architecture = self._text(data.get("revenue_architecture")) or self._deterministic_architecture(extraction, reason)

        if moat is None:
            moat = {
                "economic_moat_type": "None",
                "secondary_moat_types": [],
                "segment_specific_moats": [],
                "moat_durability": "Indeterminate",
                "moat_trajectory": "Indeterminate",
                "moat_rationale": _not_assessed(reason),
                "moat_assessment": "not_assessed",
            }

        dq.claims_verified = len({(e.claim, e.subject) for e in evidence})
        return BusinessMoatOutput(
            ticker=ticker,
            fiscal_year=fiscal_year,
            business_summary=business_summary,
            revenue_architecture=revenue_architecture,
            reportable_segments=segment_names,
            primary_product_segments=products,
            segment_details=details,
            segment_financials=extraction.segments,
            revenue_lines=extraction.revenue_lines,
            segment_table_markdown=render_segment_table_markdown(extraction.segments),
            revenue_mix_markdown=render_revenue_mix_markdown(extraction.revenue_lines),
            competitive_threats=threats,
            pricing_power_assessment=pricing,
            customer_concentration=concentration,
            evidence=evidence,
            data_quality=dq,
            citations=self._build_citations(evidence, pool, extraction, filing, items, ticker, fiscal_year),
            **moat,
        )

    # --- evidence ---------------------------------------------------------
    def _verify(
        self,
        raw_evidence: Any,
        pool: Pool,
        dq: StrategistDataQuality,
        claim: str,
        subject: Optional[str],
        supports: Optional[Callable[[str, str], bool]] = None,
    ) -> List[StrategyEvidence]:
        """Verifies up to MAX_EVIDENCE_PER_CLAIM quotes: the claimed chunk (near-verbatim), else any chunk (exact).

        `supports` additionally requires the quote to evidence the claim's mechanism (claim_relevance); a real
        but off-topic quote is counted in `quotes_off_topic` and not kept.
        """
        if isinstance(raw_evidence, dict):
            raw_evidence = [raw_evidence]
        if not isinstance(raw_evidence, list):
            return []
        verified: List[StrategyEvidence] = []
        seen = set()
        for ev in raw_evidence[:MAX_EVIDENCE_PER_CLAIM]:
            if not isinstance(ev, dict):
                continue
            quote = ev.get("quote")
            chunk_id = str(ev.get("chunk_id") or "").strip()
            if not isinstance(quote, str) or not quote.strip():
                continue
            quote = quote.strip()[:MAX_QUOTE_CHARS]
            hit = pool.get(chunk_id) if chunk_id in pool and verify_quote(quote, pool[chunk_id]["content"]) else None
            if hit is None:
                hit = next((c for cid, c in pool.items() if cid != chunk_id and verify_quote(quote, c["content"], allow_fuzzy=False)), None)
            if hit is None:
                dq.quotes_rejected += 1
                continue
            if supports is not None and not supports(quote, hit.get("content") or ""):
                dq.quotes_off_topic += 1
                continue
            key = (hit["chunk_id"], normalize_text(quote))
            if key in seen:
                continue
            seen.add(key)
            dq.quotes_verified += 1
            verified.append(
                StrategyEvidence(
                    claim=claim,  # type: ignore[arg-type]
                    subject=subject,
                    chunk_id=hit["chunk_id"],
                    quote=quote,
                    item=hit.get("item"),
                    breadcrumb=hit.get("breadcrumb"),
                )
            )
        return verified

    @staticmethod
    def _text(value: Any) -> str:
        return value.strip() if isinstance(value, str) and value.strip() else ""

    @staticmethod
    def _coerce_moat(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        v = value.strip()
        for t in MOAT_TYPES:
            if v.lower() == t.lower():
                return t
        low = v.lower()
        if low in ("none", "no moat", "n/a"):
            return "None"
        for needle, mapped in _MOAT_ALIASES:
            if needle in low:
                return mapped
        return None

    @staticmethod
    def _is_not_disclosed(text: str) -> bool:
        low = text.lower()
        return any(h in low for h in _NOT_DISCLOSED_HINTS)

    # --- moats ------------------------------------------------------------
    def _enforce_moats(
        self,
        data: Dict[str, Any],
        pool: Pool,
        segment_names: List[str],
        line_names: List[str],
        evidence: List[StrategyEvidence],
        dq: StrategistDataQuality,
    ) -> Dict[str, Any]:
        primary = data.get("primary_moat") if isinstance(data.get("primary_moat"), dict) else {}
        p_type = self._coerce_moat(primary.get("type"))
        p_rationale = self._text(primary.get("rationale"))
        if p_type and p_type != "None":
            dq.claims_proposed += 1
        p_evidence = (
            self._verify(primary.get("evidence"), pool, dq, "primary_moat", p_type, lambda q, _c: moat_quote_supports(p_type, q))
            if p_type and p_type != "None" else []
        )

        secondaries: List[Tuple[str, str, List[StrategyEvidence]]] = []
        raw_secondary = data.get("secondary_moats") if isinstance(data.get("secondary_moats"), list) else []
        for item in raw_secondary:
            if not isinstance(item, dict):
                continue
            s_type = self._coerce_moat(item.get("type"))
            if not s_type or s_type == "None":
                continue
            dq.claims_proposed += 1
            s_evidence = self._verify(
                item.get("evidence"), pool, dq, "secondary_moat", s_type, lambda q, _c, t=s_type: moat_quote_supports(t, q)
            )
            if not s_evidence:
                dq.claims_dropped += 1
                dq.notes.append(f"Secondary moat '{s_type}' dropped: no verified filing quote evidencing its mechanism.")
                continue
            secondaries.append((s_type, self._text(item.get("rationale")), s_evidence))

        assessment = "verified"
        if p_type and p_type != "None" and p_evidence:
            evidence.extend(p_evidence)
        elif secondaries:
            promoted = secondaries.pop(0)
            if p_type and p_type != "None":
                dq.claims_dropped += 1
                dq.notes.append(
                    f"Primary moat '{p_type}' had no verified quote evidencing its mechanism; verified secondary moat "
                    f"'{promoted[0]}' promoted to primary."
                )
            p_type, p_rationale, p_evidence = promoted
            p_evidence = [e.model_copy(update={"claim": "primary_moat"}) for e in p_evidence]
            evidence.extend(p_evidence)
        elif p_type == "None":
            assessment = "no_moat"
        else:
            assessment = "unsupported"
            if p_type:
                dq.claims_dropped += 1
            dq.notes.append(
                f"Primary moat {repr(p_type) if p_type else '(missing)'} had no verified filing quote evidencing its "
                "mechanism; moat downgraded to None / Indeterminate."
            )

        kept_secondary: List[str] = []
        for s_type, _, s_evidence in secondaries:
            if s_type != p_type and s_type not in kept_secondary and len(kept_secondary) < MAX_SECONDARY_MOATS:
                kept_secondary.append(s_type)
                evidence.extend(s_evidence)

        segment_moats = self._enforce_segment_moats(data, pool, segment_names, line_names, evidence, dq)

        durability = _DURABILITY.get(str(data.get("moat_durability") or "").strip().lower(), "Indeterminate")
        trajectory = _TRAJECTORY.get(str(data.get("moat_trajectory") or "").strip().lower(), "Indeterminate")
        if assessment == "verified":
            moat_quotes = {(e.chunk_id, normalize_text(e.quote)) for e in evidence if e.claim in ("primary_moat", "secondary_moat")}
            if durability == "Wide" and len(moat_quotes) < WIDE_MIN_QUOTES:
                durability = "Narrow"
                dq.notes.append(f"Durability capped at Narrow: 'Wide' needs at least {WIDE_MIN_QUOTES} distinct verified moat quotes.")
            elif durability == "None":
                durability = "Indeterminate"
                dq.notes.append("Durability 'None' contradicts a verified moat; set to Indeterminate.")
            rationale = p_rationale or f"{p_type} evidenced by the quoted filing disclosures."
        elif assessment == "no_moat":
            durability = durability if durability in ("None", "Indeterminate") else "None"
            trajectory = "Indeterminate" if trajectory == "Expanding" else trajectory
            rationale = p_rationale or "The filing evidences no structural competitive advantage."
        else:
            p_type, durability, trajectory = "None", "Indeterminate", "Indeterminate"
            kept_secondary = []
            rationale = _not_assessed("no proposed moat was supported by a verified filing quote that evidences its mechanism")

        return {
            "economic_moat_type": p_type if assessment == "verified" else "None",
            "secondary_moat_types": kept_secondary,
            "segment_specific_moats": segment_moats,
            "moat_durability": durability,
            "moat_trajectory": trajectory,
            "moat_rationale": rationale,
            "moat_assessment": assessment,
        }

    def _enforce_segment_moats(
        self,
        data: Dict[str, Any],
        pool: Pool,
        segment_names: List[str],
        line_names: List[str],
        evidence: List[StrategyEvidence],
        dq: StrategistDataQuality,
    ) -> List[SegmentMoat]:
        targets = segment_names or line_names
        lookup = {name_key(n): n for n in targets}
        raw = data.get("segment_moats") if isinstance(data.get("segment_moats"), list) else []
        kept: List[SegmentMoat] = []
        seen = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            m_type = self._coerce_moat(item.get("type"))
            name = clean_name(str(item.get("segment_name") or ""))
            if not m_type or not name:
                continue
            dq.claims_proposed += 1
            canonical = self._match_name(name, lookup) if lookup else name
            if canonical is None:
                dq.claims_dropped += 1
                dq.notes.append(f"Segment moat for '{name}' dropped: not a code-extracted segment or revenue line.")
                continue
            if name_key(canonical) in seen:
                continue
            seg_evidence = self._verify(
                item.get("evidence"), pool, dq, "segment_moat", canonical,
                # The segment must be named in the quote or in the chunk it came from.
                lambda q, c, t=m_type, n=canonical: moat_quote_supports(t, q) and segment_quote_mentions(n, f"{q} {c}"),
            )
            if not seg_evidence:
                dq.claims_dropped += 1
                dq.notes.append(
                    f"Segment moat for '{canonical}' dropped: no verified quote naming the segment and evidencing {m_type}."
                )
                continue
            seen.add(name_key(canonical))
            evidence.extend(seg_evidence)
            kept.append(
                SegmentMoat(
                    segment_name=canonical,
                    moat_type=m_type,  # type: ignore[arg-type]
                    moat_rationale=self._text(item.get("rationale")) or f"{m_type} evidenced by the quoted disclosure.",
                )
            )
        return kept

    @staticmethod
    def _match_name(name: str, lookup: Dict[str, str]) -> Optional[str]:
        key = name_key(name)
        if key in lookup:
            return lookup[key]
        candidates = [v for k, v in lookup.items() if k and (k in key or key in k)]
        return candidates[0] if len(candidates) == 1 else None

    # --- threats / pricing / concentration -------------------------------
    def _enforce_threats(
        self, data: Dict[str, Any], pool: Pool, evidence: List[StrategyEvidence], dq: StrategistDataQuality
    ) -> List[str]:
        raw = data.get("competitive_threats") if isinstance(data.get("competitive_threats"), list) else []
        kept: List[str] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            threat = self._text(item.get("threat"))
            if not threat:
                continue
            dq.claims_proposed += 1
            t_evidence = self._verify(item.get("evidence"), pool, dq, "competitive_threat", threat,
                                      lambda q, _c: threat_quote_supports(q))
            if not t_evidence or len(kept) >= MAX_THREATS:
                dq.claims_dropped += 1
                continue
            kept.append(threat)
            evidence.extend(t_evidence)
        if raw and not kept:
            dq.notes.append("No proposed competitive threat was supported by a verified filing quote.")
        return kept

    def _enforce_pricing(
        self, data: Dict[str, Any], pool: Pool, evidence: List[StrategyEvidence], dq: StrategistDataQuality
    ) -> str:
        raw = data.get("pricing_power") if isinstance(data.get("pricing_power"), dict) else {}
        assessment = self._text(raw.get("assessment"))
        if not assessment or self._is_not_disclosed(assessment):
            return NOT_DISCLOSED
        dq.claims_proposed += 1
        verified = self._verify(raw.get("evidence"), pool, dq, "pricing_power", None)
        if not verified:
            dq.claims_dropped += 1
            dq.notes.append("Pricing-power assessment dropped: no verified filing quote.")
            return NOT_DISCLOSED
        evidence.extend(verified)
        if not any(pricing_claim_supported(assessment, e.quote) for e in verified):
            dq.claims_dropped += 1
            dq.notes.append("Pricing-power claim not evidenced by its quotes (they describe pricing, not pricing power); "
                            "replaced by the verified quote.")
            return f'Filing states: "{verified[0].quote}"'
        if not numbers_supported(assessment, [pool[e.chunk_id]["content"] for e in verified]):
            dq.notes.append("Pricing-power text cited figures absent from its quotes; replaced by the verified quote.")
            return f'Filing states: "{verified[0].quote}"'
        return assessment

    def _enforce_concentration(
        self,
        data: Dict[str, Any],
        pool: Pool,
        disclosures: Sequence[DisclosureSentence],
        evidence: List[StrategyEvidence],
        dq: StrategistDataQuality,
        assessed: bool,
        reason: str,
    ) -> str:
        raw = data.get("customer_concentration") if isinstance(data.get("customer_concentration"), dict) else {}
        disclosure = self._text(raw.get("disclosure"))
        if assessed and disclosure and not self._is_not_disclosed(disclosure):
            dq.claims_proposed += 1
            verified = self._verify(raw.get("evidence"), pool, dq, "customer_concentration", None)
            if verified and numbers_supported(disclosure, [pool[e.chunk_id]["content"] for e in verified]):
                evidence.extend(verified)
                return disclosure
            dq.claims_dropped += 1
            dq.notes.append("Customer-concentration text lacked a verified quote or supported figures; replaced deterministically.")

        # Deterministic fallback: the screened filing sentence itself, verbatim.
        revenue = [d for d in disclosures if d.kind == "revenue_concentration"]
        receivables = [d for d in disclosures if d.kind == "receivables_concentration"]
        chosen = revenue[0] if revenue else receivables[0] if receivables else None
        if chosen is not None and chosen.chunk_id in pool:
            chunk = pool[chosen.chunk_id]
            evidence.append(
                StrategyEvidence(
                    claim="customer_concentration",
                    subject="code_screened",
                    chunk_id=chosen.chunk_id,
                    quote=chosen.text[:MAX_QUOTE_CHARS],
                    item=chunk.get("item"),
                    breadcrumb=chunk.get("breadcrumb"),
                )
            )
            if revenue:
                return f'Filing states: "{chosen.text}"'
            return f'{NOT_DISCLOSED} for revenue. Receivables: "{chosen.text}"'
        return NOT_DISCLOSED if assessed else _not_assessed(reason)

    # --- product lines ----------------------------------------------------
    def _reconcile_details(
        self,
        data: Dict[str, Any],
        pool: Pool,
        extraction: SegmentExtraction,
        line_names: List[str],
        segment_names: List[str],
        dq: StrategistDataQuality,
    ) -> Tuple[List[SegmentDetail], List[str]]:
        """segment_details / primary_product_segments follow the code-extracted revenue lines, else the
        (non-geographic) reportable segments.

        Without either, a model-proposed product name is kept only if it appears in the filing text.
        """
        raw_details = data.get("segment_details")
        proposed: List[Dict[str, Any]] = [d for d in raw_details if isinstance(d, dict)] if isinstance(raw_details, list) else []
        targets = line_names or (segment_names if extraction.segment_basis != "geographic" else [])
        if targets:
            lookup = {name_key(n): n for n in targets}
            by_target: Dict[str, Dict[str, Any]] = {}
            unmatched = []
            for d in proposed:
                canonical = self._match_name(str(d.get("name") or ""), lookup)
                if canonical is None:
                    unmatched.append(str(d.get("name") or ""))
                    continue
                by_target.setdefault(canonical, d)
            if unmatched:
                dq.notes.append(
                    f"{len(unmatched)} model-proposed product line(s) not in the extracted revenue table were dropped "
                    f"({', '.join(n for n in unmatched[:3] if n)})."
                )
            figures = {name_key(f.name): f for f in self._latest(extraction.revenue_lines)}
            figures.update({name_key(f.segment_name): f for f in self._latest(extraction.segments)})
            details = []
            for name in targets:
                d = by_target.get(name, {})
                details.append(
                    SegmentDetail(
                        name=name,
                        description=self._text(d.get("description")) or self._figure_description(name, figures.get(name_key(name))),
                        growth_drivers=self._drivers(d.get("growth_drivers")),
                    )
                )
            return details, list(targets)

        corpus = normalize_text(" ".join(c.get("content", "") for c in pool.values()))
        details = []
        for d in proposed:
            name = clean_name(str(d.get("name") or ""))
            if not name or normalize_text(name) not in corpus:
                if name:
                    dq.notes.append(f"Product line '{name}' dropped: not named in the filing text.")
                continue
            details.append(
                SegmentDetail(name=name, description=self._text(d.get("description")) or name, growth_drivers=self._drivers(d.get("growth_drivers")))
            )
        return details, [d.name for d in details]

    @staticmethod
    def _drivers(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [v.strip() for v in value if isinstance(v, str) and v.strip()][:4]

    @staticmethod
    def _figure_description(name: str, figure: Any) -> str:
        if figure is None:
            return f"{name} (reported revenue line)."
        share = f", {figure.revenue_share_pct:.1f}% of revenue" if figure.revenue_share_pct is not None else ""
        return f"{name}: FY{figure.fiscal_year} revenue ${figure.revenue:,.0f}M{share} (from the 10-K tables)."

    def _deterministic_architecture(self, extraction: SegmentExtraction, reason: str) -> str:
        lines = [f for f in self._latest(extraction.revenue_lines) if f.parent is None]
        if not lines:
            return _not_assessed(reason)
        mix = ", ".join(
            f"{f.name} {f.revenue_share_pct:.1f}%" for f in lines if f.revenue_share_pct is not None
        )
        return f"Revenue mix by product / service line (FY{lines[0].fiscal_year}, from the 10-K revenue disaggregation): {mix}."

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
