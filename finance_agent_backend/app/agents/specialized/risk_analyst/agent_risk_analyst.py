"""Risk & Red Flag Analyst Agent (evidence-first).

Code reads the complete Item 1A of the target 10-K (and the prior-year Item 1A for a
disclosure diff) before the LLM runs. The LLM proposes a ranked risk inventory with verbatim
quotes and likelihood / impact ratings; code then verifies every quote against the retrieved
chunks, drops unsupported claims, applies materiality and boilerplate screens, merges
duplicates, derives severity from a fixed likelihood x impact matrix, re-points the primary
threat when needed, and renders the verified risk matrix and evidence table.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.risk_analyst.state_risk_analyst import (
    FINANCIAL_TRANSMISSIONS,
    RISK_CATEGORIES,
    RiskAuditOutput,
    RiskDataQuality,
    RiskEvidence,
    RiskItem,
)
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool
from app.agents.tools.risk_disclosure_tools import (
    LEVEL_ORDER,
    QUANTIFIED_FIGURE,
    RISK_SECTION_ITEM,
    SEVERITY_ORDER,
    build_prior_index,
    build_section_context,
    classify_disclosure_change,
    derive_overall_profile,
    derive_severity,
    diff_risk_disclosures,
    disclosure_covered,
    extract_quantified_disclosures,
    is_boilerplate,
    is_material_disclosure,
    locate_section_chunks,
    normalize_level,
    normalize_text,
    numbers_supported,
    render_risk_evidence_table,
    render_risk_matrix_markdown,
    section_not_provided,
    states_immaterial,
    substantiates_high_impact,
    verify_quote,
)

logger = logging.getLogger("finance_agent.agents.risk_analyst")

MAX_RISKS = 8
MIN_RISKS = 5
MAX_QUOTE_CHARS = 300
MAX_SIGNPOSTS = 3
DUPLICATE_TITLE_JACCARD = 0.5
DUPLICATE_TITLE_JACCARD_SHARED_CHUNK = 0.3

_TITLE_STOPWORDS = {
    "risk", "risks", "the", "and", "of", "to", "in", "on", "for", "our", "from", "with",
    "related", "due", "its", "their", "a", "an", "by", "or", "potential", "exposure",
}

_CATEGORY_KEYWORDS: List[Tuple[str, str]] = [
    (r"regulat|legal|litigation|lawsuit|antitrust|export\s+control|compliance|government|sanction", "Regulatory & Legal"),
    (r"cyber|technolog|privacy|intellectual\s+property|\bip\b|obsolesc|software|security\s+breach", "Technological & Cybersecurity"),
    (r"supply|supplier|manufactur|foundry|concentrat|customer|single.source|sole.source", "Supply Chain & Concentration"),
    (r"compet|demand|pricing|market\s+share|substitut|cyclical", "Competitive & Demand"),
    (r"macro|geopolit|currency|foreign\s+exchange|tariff|trade|interest\s+rate|inflation|recession", "Macroeconomic & Geopolitical"),
    (r"debt|liquidity|capital\s+structure|financing|credit|leverage|tax|impairment", "Financial & Capital Structure"),
]

_NOTE_IMMATERIAL = "Impact capped at Medium: the cited filing text states the matter is not expected to be material."
_NOTE_LEGAL_OUTSIDE = "Impact capped at Medium: legal matter sourced outside Item 1A without a quantified exposure."
_NOTE_BOILERPLATE = "Impact capped at Low: generic boilerplate risk without a company-specific exposure."
_NOTE_UNSUBSTANTIATED = "Impact capped at Medium: no filing-stated magnitude or realized effect supports a High impact."
_NOTE_EXPOSURE_UNIT = "Quantified exposure removed: not stated as a percentage or currency amount."
MAX_UNCOVERED_LISTED = 5
MAX_REPAIR_DISCLOSURES = 4
_NOTE_EXPOSURE = "Quantified exposure removed: its figures do not appear in the cited filing text."
_NOTE_MITIGATION = "Mitigation removed: no verified filing quote supports it."


def _rank_key(risk: RiskItem) -> Tuple[int, int, int]:
    return SEVERITY_ORDER[risk.severity], LEVEL_ORDER[risk.impact], LEVEL_ORDER[risk.likelihood]


def _higher(a: str, b: str) -> str:
    return a if LEVEL_ORDER[a] <= LEVEL_ORDER[b] else b


def _title_tokens(title: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", normalize_text(title)) if len(w) > 2 and w not in _TITLE_STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@AgentRegistry.register("risk_analyst")
class RiskAnalystAgent(StructuredAgent[RiskAuditOutput]):
    """Evidence-first qualitative agent analyzing Item 1A risk disclosures and existential threats."""

    prompt_name = "risk_analyst"
    tools = [retrieve_10k_narrative_tool]
    output_schema = RiskAuditOutput
    default_recursion_limit = 25

    def __init__(self, model_name: str = "openai:gpt-4o-mini", recursion_limit: Optional[int] = None):
        super().__init__(model_name=model_name, recursion_limit=recursion_limit)
        self._run: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze(
        self,
        ticker: str,
        fiscal_year: int,
        available_fiscal_years: Optional[List[int]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> RiskAuditOutput:
        """Analyzes the 10-K Item 1A for `ticker` / `fiscal_year` and returns a verified RiskAuditOutput."""
        ticker = ticker.upper()
        current, located_by = self._safe_locate(ticker, fiscal_year)
        if located_by != "not_found" and section_not_provided(current):
            return self._not_provided_output(ticker, fiscal_year, current, located_by)

        prior_year = self._resolve_prior_year(fiscal_year, available_fiscal_years)
        prior = self._safe_locate(ticker, prior_year)[0] if prior_year else []
        if prior and section_not_provided(prior):
            prior = []

        if current:
            context, mode, chars = build_section_context(current)
        else:
            context, mode, chars = "", "rag_fallback", 0

        diff = None
        if current and prior:
            try:
                diff = diff_risk_disclosures(current, prior, prior_year)
            except Exception as e:
                logger.warning(f"Item 1A diff failed for {ticker} FY{fiscal_year} vs FY{prior_year}: {e}")

        quantified = extract_quantified_disclosures(current)
        data_quality = RiskDataQuality(
            coverage_mode=mode,
            section_located_by=located_by,
            item1a_chunks=len(current),
            chars_read=chars,
            prior_fiscal_year=prior_year if diff else None,
            quantified_disclosures=len(quantified),
        )
        if mode == "rag_fallback":
            data_quality.notes.append(
                f"Item 1A could not be located in the {ticker} FY{fiscal_year} 10-K (no section label or heading); "
                "risks were sourced through hybrid retrieval across the filing and may be incomplete."
            )
        if located_by == "heading_boundary":
            data_quality.notes.append(
                "Item 1A was located from the filing's own section headings because the stored section labels were wrong; "
                "consider re-ingesting this filing."
            )
        elif mode == "budgeted_section":
            data_quality.notes.append("Item 1A exceeded the context budget; non-summary paragraphs were truncated to their lead text.")
        if prior_year and not diff:
            data_quality.notes.append(f"Prior-year (FY{prior_year}) Item 1A unavailable; year-over-year change not assessed.")

        self._run = {
            "pool": {c.chunk_id: c.model_dump() for c in current},
            "prior_index": build_prior_index(prior) if diff else None,
            "diff": diff,
            "quantified": quantified,
            "data_quality": data_quality,
            "llm_output_missing": False,
        }
        logger.info(
            f"RiskAnalyst {ticker} FY{fiscal_year}: mode={mode}, chunks={len(current)}, chars={chars}, "
            f"prior={prior_year if diff else None}"
        )

        query = render_prompt(
            "prompt_risk_analyst_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            coverage_mode=mode,
            section_context=context,
            diff=diff,
            quantified=quantified,
        )
        output = self.execute_structured(query, ticker=ticker, fiscal_year=fiscal_year, callbacks=callbacks)
        return self._repair_coverage(output, ticker, fiscal_year, callbacks)

    # ------------------------------------------------------------------
    # Coverage repair pass
    # ------------------------------------------------------------------
    def _repair_coverage(
        self,
        output: RiskAuditOutput,
        ticker: str,
        fiscal_year: int,
        callbacks: Optional[List[Any]],
    ) -> RiskAuditOutput:
        """Asks the model, in a small focused call, for risks covering material checklist figures it left uncited.

        The proposed risks are added to the model's original proposals and the whole inventory is
        re-enforced, so repaired risks pass exactly the same verification, screens and ranking.
        """
        run = self._ensure_run()
        targets = [d for d in run.get("uncovered", []) if is_material_disclosure(d.text)][:MAX_REPAIR_DISCLOSURES]
        snapshot = run.get("pre_enforcement")
        if not targets or snapshot is None:
            return output

        pool = run["pool"]
        chunk_ids = list(dict.fromkeys(d.chunk_id for d in targets))
        prompt = render_prompt(
            "prompt_risk_analyst_repair.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            targets=targets,
            chunks=[pool[cid] for cid in chunk_ids if cid in pool],
            existing_titles=[r.risk_title for r in output.identified_risks],
        )
        try:
            raw = self.parse_json_content(self._run_repair_llm(prompt, callbacks))
        except Exception as e:
            logger.warning(f"Coverage repair pass failed for {ticker} FY{fiscal_year}: {e}")
            output.data_quality.notes.append("Coverage repair pass failed; uncited quantified disclosures remain.")
            return output

        proposals = raw.get("identified_risks") if isinstance(raw.get("identified_risks"), list) else []
        repaired = []
        for idx, item in enumerate(proposals):
            sanitized = self._sanitize_risk(item, idx)
            if sanitized:
                sanitized["risk_id"] = f"C{idx + 1}"
                repaired.append(RiskItem.model_validate(sanitized))
        if not repaired:
            output.data_quality.notes.append("Coverage repair pass proposed no usable risks; uncited quantified disclosures remain.")
            return output

        base_output, result = snapshot
        combined = base_output.model_copy(deep=True)
        combined.identified_risks = list(combined.identified_risks) + repaired
        run["data_quality"].risks_proposed += len(proposals)
        run["data_quality"].notes.append(
            f"Coverage repair pass proposed {len(repaired)} risk(s) for {len(targets)} uncited quantified disclosure(s)."
        )
        return self._post_process_output(combined, result)

    def _run_repair_llm(self, prompt: str, callbacks: Optional[List[Any]]) -> str:
        """Single tool-free completion for the coverage repair pass."""
        from dotenv import load_dotenv
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        load_dotenv(override=True)
        llm = ChatOpenAI(model=self.model_name.replace("openai:", ""), temperature=0, max_retries=5)
        config = {"callbacks": callbacks} if callbacks else {}
        return llm.invoke([HumanMessage(content=prompt)], config=config).content

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_prior_year(fiscal_year: int, available_fiscal_years: Optional[List[int]]) -> Optional[int]:
        if available_fiscal_years is None:
            return fiscal_year - 1
        prior = [y for y in available_fiscal_years if y < fiscal_year]
        return max(prior) if prior else None

    @staticmethod
    def _safe_locate(ticker: str, fiscal_year: int) -> Tuple[List[Any], str]:
        try:
            return locate_section_chunks(ticker, fiscal_year, RISK_SECTION_ITEM)
        except Exception as e:
            logger.warning(f"Item 1A pre-fetch failed for {ticker} FY{fiscal_year}: {e}")
            return [], "not_found"

    def _not_provided_output(
        self, ticker: str, fiscal_year: int, chunks: List[Any], located_by: str
    ) -> RiskAuditOutput:
        """Item 1A exists but discloses no risk factors (e.g. smaller reporting company): rate nothing, call no LLM."""
        statement = " ".join(c.content for c in chunks).strip()
        citations = [
            {k: v for k, v in c.model_dump().items() if k in ("chunk_id", "ticker", "fiscal_year", "item", "breadcrumb", "sub_section")}
            for c in chunks
        ]
        note = (
            f"The FY{fiscal_year} 10-K provides no Item 1A risk factors"
            + (f' (filing states: "{statement[:200]}")' if statement else "")
            + "; no risks were rated."
        )
        self._run = {}
        logger.info(f"RiskAnalyst {ticker} FY{fiscal_year}: Item 1A not provided; LLM skipped")
        return RiskAuditOutput(
            ticker=ticker,
            fiscal_year=fiscal_year,
            data_quality=RiskDataQuality(
                coverage_mode="not_provided",
                section_located_by=located_by,
                item1a_chunks=len(chunks),
                chars_read=len(statement),
                notes=[note],
            ),
            risk_matrix_markdown="_The filing provides no Item 1A risk factors (not applicable / smaller reporting company); no risks were rated._",
            evidence_table_markdown=render_risk_evidence_table(citations),
            citations=citations,
        )

    def _ensure_run(self) -> Dict[str, Any]:
        if not self._run:
            self._run = {
                "pool": {},
                "prior_index": None,
                "diff": None,
                "quantified": [],
                "data_quality": RiskDataQuality(),
                "llm_output_missing": False,
            }
        return self._run

    # ------------------------------------------------------------------
    # Stage 1: sanitize raw LLM output so it validates
    # ------------------------------------------------------------------
    def _pre_validate_data(self, data: Dict[str, Any], result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        run = self._ensure_run()
        raw = data.get("identified_risks")
        if not isinstance(raw, list):
            run["llm_output_missing"] = True
            raw = []
        sanitized = [r for r in (self._sanitize_risk(item, idx) for idx, item in enumerate(raw)) if r]
        dq: RiskDataQuality = run["data_quality"]
        dq.risks_proposed = len(raw)
        if len(sanitized) < len(raw):
            dq.notes.append(f"{len(raw) - len(sanitized)} malformed risk entries discarded (missing title or summary).")

        primary_id = data.get("primary_threat_risk_id")
        threat = data.get("primary_existential_threat")
        return {
            "ticker": data.get("ticker"),
            "fiscal_year": data.get("fiscal_year"),
            "identified_risks": sanitized,
            "primary_threat_risk_id": str(primary_id) if primary_id not in (None, "") else None,
            "primary_existential_threat": threat.strip() if isinstance(threat, str) else "",
        }

    def _sanitize_risk(self, item: Any, idx: int) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        title = str(item.get("risk_title") or "").strip()
        summary = str(item.get("risk_summary") or "").strip()
        if not title or not summary:
            return None

        adjustments: List[str] = []
        likelihood = normalize_level(item.get("likelihood"))
        impact = normalize_level(item.get("impact"))
        if likelihood is None:
            likelihood = "Medium"
            adjustments.append("Likelihood not rated by the model; defaulted to Medium.")
        if impact is None:
            impact = "Medium"
            adjustments.append("Impact not rated by the model; defaulted to Medium.")

        transmissions = item.get("financial_transmission") or []
        if isinstance(transmissions, str):
            transmissions = [transmissions]
        lookup = {t.lower(): t for t in FINANCIAL_TRANSMISSIONS}
        transmission = list(dict.fromkeys(
            lookup[t.strip().lower()] for t in transmissions if isinstance(t, str) and t.strip().lower() in lookup
        ))

        signposts = item.get("monitoring_signposts") or []
        if isinstance(signposts, str):
            signposts = [signposts]

        return {
            "risk_id": str(item.get("risk_id") or f"R{idx + 1}"),
            "risk_category": self._coerce_category(item.get("risk_category"), f"{title} {summary}"),
            "risk_title": title,
            "risk_summary": summary,
            "likelihood": likelihood,
            "impact": impact,
            "severity": derive_severity(likelihood, impact),
            "financial_transmission": transmission,
            "quantified_exposure": self._clean_str(item.get("quantified_exposure")),
            "mitigating_factors": self._clean_str(item.get("mitigating_factors")),
            "mitigation_evidence": self._sanitize_evidence(item.get("mitigation_evidence")),
            "evidence": [e for e in (self._sanitize_evidence(x) for x in (item.get("evidence") or [])) if e],
            "monitoring_signposts": [s.strip() for s in signposts if isinstance(s, str) and s.strip()][:MAX_SIGNPOSTS],
            "adjustments": adjustments,
        }

    @staticmethod
    def _clean_str(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        v = value.strip()
        return None if not v or v.lower() in ("null", "none", "n/a", "not disclosed") else v

    @staticmethod
    def _sanitize_evidence(value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        chunk_id, quote = value.get("chunk_id"), value.get("quote")
        if not isinstance(chunk_id, str) or not isinstance(quote, str) or not quote.strip():
            return None
        return {"chunk_id": chunk_id.strip(), "quote": quote.strip()[:MAX_QUOTE_CHARS]}

    @staticmethod
    def _coerce_category(category: Any, fallback_text: str) -> str:
        if isinstance(category, str):
            for c in RISK_CATEGORIES:
                if category.strip().lower() == c.lower():
                    return c
        for text in (category if isinstance(category, str) else "", fallback_text):
            for pattern, mapped in _CATEGORY_KEYWORDS:
                if text and re.search(pattern, text, re.IGNORECASE):
                    return mapped
        return "Operational"

    # ------------------------------------------------------------------
    # Stage 2: deterministic enforcement on the validated model
    # ------------------------------------------------------------------
    def _post_process_output(self, output: RiskAuditOutput, result: Dict[str, Any], **kwargs) -> RiskAuditOutput:
        run = self._ensure_run()
        run["pre_enforcement"] = (output.model_copy(deep=True), result)
        dq: RiskDataQuality = run["data_quality"].model_copy(deep=True)
        if run["llm_output_missing"]:
            dq.notes.append("llm_output_missing: the model returned no parsable risk inventory.")

        pool: Dict[str, Dict[str, Any]] = dict(run["pool"])
        for cid, chunk in self._tool_chunks(result.get("messages", [])).items():
            pool.setdefault(cid, chunk)

        risks = [r.model_copy(deep=True) for r in output.identified_risks]
        if not dq.risks_proposed:
            dq.risks_proposed = len(risks)
        by_llm_id = {r.risk_id: r for r in risks}
        chosen = by_llm_id.get(output.primary_threat_risk_id or "")

        verified: List[RiskItem] = []
        for risk in risks:
            if self._enforce_risk(risk, pool, dq):
                verified.append(risk)
            else:
                dq.risks_dropped_unverified += 1

        survivors, absorbed = self._merge_duplicates(verified)
        dq.risks_merged = len(absorbed)
        ranked = sorted(survivors, key=_rank_key)
        if len(ranked) > MAX_RISKS:
            dq.notes.append(f"{len(ranked) - MAX_RISKS} lower-ranked verified risks omitted (cap of {MAX_RISKS}).")
            ranked = ranked[:MAX_RISKS]
        if len(ranked) < MIN_RISKS:
            dq.notes.append(f"Only {len(ranked)} risks survived evidence verification (target {MIN_RISKS}-{MAX_RISKS}).")

        for i, risk in enumerate(ranked, start=1):
            risk.risk_id = f"R{i}"
            risk.disclosure_change = classify_disclosure_change(
                [e.quote for e in risk.evidence if e.item == RISK_SECTION_ITEM], run["prior_index"]
            )

        output.identified_risks = ranked
        self._check_quantified_coverage(ranked, run["quantified"], dq, run)
        self._resolve_primary_threat(output, chosen, absorbed, dq)
        output.overall_risk_profile = derive_overall_profile([r.severity for r in ranked])
        output.citations = self._build_citations(ranked, pool)
        output.risk_matrix_markdown = render_risk_matrix_markdown(ranked)
        output.evidence_table_markdown = render_risk_evidence_table(output.citations)
        output.disclosure_changes = run["diff"]
        output.data_quality = dq
        return output

    def _enforce_risk(self, risk: RiskItem, pool: Dict[str, Dict[str, Any]], dq: RiskDataQuality) -> bool:
        """Verifies evidence and applies screens in place. Returns False if the risk has no verified evidence."""
        kept: List[RiskEvidence] = []
        seen = set()
        for ev in risk.evidence:
            hit = self._verify_evidence(ev, pool)
            if hit is None:
                dq.quotes_rejected += 1
                continue
            key = (hit.chunk_id, normalize_text(hit.quote))
            if key not in seen:
                seen.add(key)
                kept.append(hit)
        if not kept:
            return False
        risk.evidence = kept
        chunk_texts = [pool[e.chunk_id]["content"] for e in kept]

        if risk.quantified_exposure and not QUANTIFIED_FIGURE.search(risk.quantified_exposure):
            risk.quantified_exposure = None
            risk.adjustments.append(_NOTE_EXPOSURE_UNIT)
        elif risk.quantified_exposure and not numbers_supported(risk.quantified_exposure, chunk_texts):
            risk.quantified_exposure = None
            risk.adjustments.append(_NOTE_EXPOSURE)

        mitigation = self._verify_evidence(risk.mitigation_evidence, pool) if risk.mitigation_evidence else None
        if risk.mitigating_factors and mitigation is None:
            risk.mitigating_factors = None
            risk.adjustments.append(_NOTE_MITIGATION)
        risk.mitigation_evidence = mitigation if risk.mitigating_factors else None

        outside = [e for e in kept if e.item != RISK_SECTION_ITEM]
        immaterial = any(states_immaterial(e.quote) for e in kept) or any(
            states_immaterial(pool[e.chunk_id]["content"]) for e in outside
        )
        if risk.impact == "High" and immaterial:
            risk.impact = "Medium"
            risk.adjustments.append(_NOTE_IMMATERIAL)
        elif (
            risk.impact == "High"
            and risk.risk_category == "Regulatory & Legal"
            and len(outside) == len(kept)
            and not risk.quantified_exposure
        ):
            risk.impact = "Medium"
            risk.adjustments.append(_NOTE_LEGAL_OUTSIDE)

        if risk.impact == "High" and not substantiates_high_impact([e.quote for e in kept], risk.quantified_exposure):
            risk.impact = "Medium"
            risk.adjustments.append(_NOTE_UNSUBSTANTIATED)

        if risk.impact != "Low" and not risk.quantified_exposure and is_boilerplate(f"{risk.risk_title} {risk.risk_summary}"):
            risk.impact = "Low"
            risk.adjustments.append(_NOTE_BOILERPLATE)

        risk.severity = derive_severity(risk.likelihood, risk.impact)
        return True

    @staticmethod
    def _verify_evidence(ev: Optional[RiskEvidence], pool: Dict[str, Dict[str, Any]]) -> Optional[RiskEvidence]:
        """Matches a quote to its claimed chunk (exact or near-verbatim), else to any pooled chunk (exact)."""
        if ev is None or not pool:
            return None
        claimed = pool.get(ev.chunk_id)
        hit = claimed if claimed and verify_quote(ev.quote, claimed["content"]) else None
        if hit is None:
            hit = next(
                (c for cid, c in pool.items() if cid != ev.chunk_id and verify_quote(ev.quote, c["content"], allow_fuzzy=False)),
                None,
            )
        if hit is None:
            return None
        return RiskEvidence(
            chunk_id=hit["chunk_id"],
            quote=ev.quote,
            item=hit.get("item"),
            breadcrumb=hit.get("breadcrumb"),
            verified=True,
        )

    @staticmethod
    def _tool_chunks(messages: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
        """Collects chunks (with content) returned by narrative-retrieval tool calls."""
        chunks: Dict[str, Dict[str, Any]] = {}
        for msg in messages:
            raw = getattr(msg, "content", "")
            if not isinstance(raw, str) or '"chunk_id"' not in raw or '"content"' not in raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            for item in parsed if isinstance(parsed, list) else [parsed]:
                if isinstance(item, dict) and item.get("chunk_id") and isinstance(item.get("content"), str):
                    chunks.setdefault(str(item["chunk_id"]), item)
        return chunks

    def _merge_duplicates(self, risks: List[RiskItem]) -> Tuple[List[RiskItem], Dict[int, RiskItem]]:
        """Folds duplicate risks into the higher-ranked one. Returns (survivors, {id(absorbed): survivor})."""
        survivors: List[RiskItem] = []
        absorbed: Dict[int, RiskItem] = {}
        for risk in sorted(risks, key=_rank_key):
            base = next((s for s in survivors if self._is_duplicate(s, risk)), None)
            if base is None:
                survivors.append(risk)
                continue
            self._absorb(base, risk)
            absorbed[id(risk)] = base
        return survivors, absorbed

    @staticmethod
    def _is_duplicate(a: RiskItem, b: RiskItem) -> bool:
        similarity = _jaccard(_title_tokens(a.risk_title), _title_tokens(b.risk_title))
        if similarity >= DUPLICATE_TITLE_JACCARD:
            return True
        shared = {e.chunk_id for e in a.evidence} & {e.chunk_id for e in b.evidence}
        return bool(shared) and similarity >= DUPLICATE_TITLE_JACCARD_SHARED_CHUNK

    @staticmethod
    def _absorb(base: RiskItem, other: RiskItem) -> None:
        seen = {(e.chunk_id, normalize_text(e.quote)) for e in base.evidence}
        for e in other.evidence:
            key = (e.chunk_id, normalize_text(e.quote))
            if key not in seen:
                seen.add(key)
                base.evidence.append(e)
        base.likelihood = _higher(base.likelihood, other.likelihood)
        base.impact = _higher(base.impact, other.impact)
        base.severity = derive_severity(base.likelihood, base.impact)
        base.financial_transmission = list(dict.fromkeys(base.financial_transmission + other.financial_transmission))
        base.monitoring_signposts = list(dict.fromkeys(base.monitoring_signposts + other.monitoring_signposts))[:MAX_SIGNPOSTS]
        if not base.quantified_exposure and other.quantified_exposure:
            base.quantified_exposure = other.quantified_exposure
        if not base.mitigating_factors and other.mitigating_factors:
            base.mitigating_factors = other.mitigating_factors
            base.mitigation_evidence = other.mitigation_evidence
        base.adjustments.append(f"Merged duplicate risk: '{other.risk_title}'.")

    @staticmethod
    def _check_quantified_coverage(
        risks: Sequence[RiskItem], quantified: Sequence[Any], dq: RiskDataQuality, run_state: Dict[str, Any]
    ) -> None:
        """Records checklist disclosures (stated figures) that no verified risk cites."""
        quotes = [e.quote for r in risks for e in r.evidence]
        quotes += [r.mitigation_evidence.quote for r in risks if r.mitigation_evidence]
        uncovered = [d for d in quantified if not disclosure_covered(d.text, quotes)]
        run_state["uncovered"] = uncovered
        dq.quantified_uncovered = [d.text for d in uncovered[:MAX_UNCOVERED_LISTED]]
        if uncovered:
            dq.notes.append(
                f"{len(uncovered)} of {len(quantified)} quantified Item 1A disclosures are not cited by any verified risk."
            )

    @staticmethod
    def _resolve_primary_threat(
        output: RiskAuditOutput,
        chosen: Optional[RiskItem],
        absorbed: Dict[int, RiskItem],
        dq: RiskDataQuality,
    ) -> None:
        ranked = output.identified_risks
        if not ranked:
            output.primary_threat_risk_id = None
            output.primary_existential_threat = ""
            return
        if chosen is not None:
            chosen = absorbed.get(id(chosen), chosen)
        top_severity = ranked[0].severity
        if chosen is not None and any(r is chosen for r in ranked) and chosen.severity == top_severity:
            output.primary_threat_risk_id = chosen.risk_id
            if not output.primary_existential_threat:
                output.primary_existential_threat = f"{chosen.risk_title}: {chosen.risk_summary}"
            return

        top = ranked[0]
        if chosen is not None:
            dq.notes.append(
                f"Primary threat re-pointed to {top.risk_id} ('{top.risk_title}'): the model's choice "
                f"('{chosen.risk_title}') was not among the most severe verified risks."
            )
        elif output.primary_threat_risk_id:
            dq.notes.append(f"Primary threat re-pointed to {top.risk_id}: the model's choice did not survive verification.")
        output.primary_threat_risk_id = top.risk_id
        output.primary_existential_threat = f"{top.risk_title}: {top.risk_summary}"

    @staticmethod
    def _build_citations(risks: Sequence[RiskItem], pool: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        seen = set()
        for risk in risks:
            evidence = list(risk.evidence) + ([risk.mitigation_evidence] if risk.mitigation_evidence else [])
            for ev in evidence:
                if ev.chunk_id in seen:
                    continue
                seen.add(ev.chunk_id)
                chunk = pool.get(ev.chunk_id, {})
                citation = {"chunk_id": ev.chunk_id}
                for key in ("ticker", "fiscal_year", "item", "breadcrumb", "sub_section"):
                    if chunk.get(key) is not None:
                        citation[key] = chunk[key]
                citations.append(citation)
        return citations
