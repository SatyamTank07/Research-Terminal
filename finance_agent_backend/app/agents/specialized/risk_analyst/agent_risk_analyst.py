"""Risk & Red Flag Analyst Agent (evidence-first).

Code reads the complete Item 1A of the target 10-K (and the prior-year Item 1A for a
disclosure diff) before the LLM runs. The LLM proposes a ranked risk inventory with verbatim
quotes and likelihood / impact ratings; code then verifies every quote against the retrieved
chunks, drops unsupported claims, applies materiality and boilerplate screens, merges
duplicates, derives severity from a fixed likelihood x impact matrix, re-points the primary
threat when needed, and renders the verified risk matrix and evidence table.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.filing.narrative_search import retrieve_10k_narrative_tool
from app.agents.shared.filing.sections import build_section_context
from app.agents.shared.llm_call import complete_once
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.risk_analyst.assemble import render_risk_evidence_table
from app.agents.specialized.risk_analyst.context import (
    RISK_SECTION_ITEM,
    coverage_notes,
    locate_item1a,
    resolve_prior_year,
    section_not_provided,
    tool_chunks,
)
from app.agents.specialized.risk_analyst.modules.disclosure_changes import (
    build_prior_index,
    classify_disclosure_change,
    diff_risk_disclosures,
)
from app.agents.specialized.risk_analyst.modules.quantified_exposures import (
    check_quantified_coverage,
    extract_quantified_disclosures,
    repair_targets,
)
from app.agents.specialized.risk_analyst.modules.risk_inventory import sanitize_risk, verify_inventory
from app.agents.specialized.risk_analyst.modules.risk_profile import (
    derive_overall_profile,
    rank_risks,
    render_risk_matrix_markdown,
    resolve_primary_threat,
)
from app.agents.specialized.risk_analyst.state_risk_analyst import (
    RiskAuditOutput,
    RiskDataQuality,
    RiskItem,
)

logger = logging.getLogger("finance_agent.agents.risk_analyst")


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
        current, located_by = locate_item1a(ticker, fiscal_year)
        if located_by != "not_found" and section_not_provided(current):
            return self._not_provided_output(ticker, fiscal_year, current, located_by)

        prior_year = resolve_prior_year(fiscal_year, available_fiscal_years)
        prior = locate_item1a(ticker, prior_year)[0] if prior_year else []
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
        data_quality.notes.extend(coverage_notes(ticker, fiscal_year, mode, located_by, prior_year, diff is not None))

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
            "risk_analyst", "prompt_risk_analyst_query.j2",
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
        targets = repair_targets(run.get("uncovered", []))
        snapshot = run.get("pre_enforcement")
        if not targets or snapshot is None:
            return output

        pool = run["pool"]
        chunk_ids = list(dict.fromkeys(d.chunk_id for d in targets))
        prompt = render_prompt(
            "risk_analyst", "prompt_risk_analyst_repair.j2",
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
            sanitized = sanitize_risk(item, idx)
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
        return complete_once(self.model_name, prompt, max_retries=5, callbacks=callbacks)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
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
        sanitized = [r for r in (sanitize_risk(item, idx) for idx, item in enumerate(raw)) if r]
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
        for cid, chunk in tool_chunks(result.get("messages", [])).items():
            pool.setdefault(cid, chunk)

        risks = [r.model_copy(deep=True) for r in output.identified_risks]
        if not dq.risks_proposed:
            dq.risks_proposed = len(risks)
        by_llm_id = {r.risk_id: r for r in risks}
        chosen = by_llm_id.get(output.primary_threat_risk_id or "")

        inventory = verify_inventory(risks, pool)
        dq.quotes_rejected += inventory.quotes_rejected
        dq.risks_dropped_unverified += inventory.dropped_unverified
        survivors, absorbed = inventory.survivors, inventory.absorbed
        dq.risks_merged = len(absorbed)
        profile = rank_risks(survivors)
        ranked = profile.risks
        dq.notes.extend(profile.notes)

        for risk in ranked:
            risk.disclosure_change = classify_disclosure_change(
                [e.quote for e in risk.evidence if e.item == RISK_SECTION_ITEM], run["prior_index"]
            )

        output.identified_risks = ranked
        self._check_quantified_coverage(ranked, run["quantified"], dq, run)
        threat = resolve_primary_threat(
            ranked, chosen, absorbed, output.primary_threat_risk_id, output.primary_existential_threat
        )
        output.primary_threat_risk_id = threat.risk_id
        output.primary_existential_threat = threat.statement
        if threat.note:
            dq.notes.append(threat.note)
        output.overall_risk_profile = derive_overall_profile([r.severity for r in ranked])
        output.citations = self._build_citations(ranked, pool)
        output.risk_matrix_markdown = render_risk_matrix_markdown(ranked)
        output.evidence_table_markdown = render_risk_evidence_table(output.citations)
        output.disclosure_changes = run["diff"]
        output.data_quality = dq
        return output

    @staticmethod
    def _check_quantified_coverage(
        risks: Sequence[RiskItem], quantified: Sequence[Any], dq: RiskDataQuality, run_state: Dict[str, Any]
    ) -> None:
        """Records checklist disclosures (stated figures) that no verified risk cites."""
        quotes = [e.quote for r in risks for e in r.evidence]
        quotes += [r.mitigation_evidence.quote for r in risks if r.mitigation_evidence]
        coverage = check_quantified_coverage(quantified, quotes)
        run_state["uncovered"] = coverage.uncovered
        dq.quantified_uncovered = list(coverage.listed)
        if coverage.note:
            dq.notes.append(coverage.note)

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


def analyze_risks(run: RunContext, upstream: Upstream) -> RiskAuditOutput:
    """Pipeline entry: verified Item 1A risk inventory with the prior-year comparison."""
    return RiskAnalystAgent().analyze(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        available_fiscal_years=list(run.available_fiscal_years),
        callbacks=run.callbacks,
    )


def summarize_risks(risk: Optional[RiskAuditOutput]) -> StatusLine:
    """SSE status line summarizing the verified risk inventory."""
    if risk is None:
        return "Analyzed Item 1A risks & threats (no risk output)", None
    if risk.data_quality.coverage_mode == "not_provided":
        return "Analyzed Item 1A risks & threats (filing provides no risk factors; nothing rated)", None
    severe = sum(1 for r in risk.identified_risks if r.severity == "Severe")
    coverage = risk.data_quality.coverage_mode.replace("_", "-")
    if risk.data_quality.section_located_by == "heading_boundary":
        coverage += " (located by headings)"
    if risk.disclosure_changes:
        new_cnt = sum(1 for r in risk.identified_risks if r.disclosure_change in ("New", "Expanded"))
        change = f"{new_cnt} new/expanded vs FY{risk.disclosure_changes.prior_fiscal_year}"
    else:
        change = "no prior-year baseline"
    return (
        f"Analyzed Item 1A risks & threats ({len(risk.identified_risks)} verified risks, {severe} severe; "
        f"{coverage} read; {change})",
        None,
    )
