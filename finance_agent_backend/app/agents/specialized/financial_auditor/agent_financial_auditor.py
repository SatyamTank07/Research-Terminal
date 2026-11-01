"""Financial Auditor & Statement Analyst Agent.

Deterministic-first audit of 10-K financial statements (Item 8):

1. Code pre-fetches the income statement, balance sheet and cash flow tables for the target
   filing (plus up to two prior filings), parses them, maps line items with generic US-GAAP
   synonyms, checks accounting identities and computes a baseline audit — no LLM involved.
2. The LLM reviews the baseline against the clean tables, may submit cited corrections and
   normalization adjustments through `refine_financial_audit_tool`, reviews restatements and
   writes the quantified auditor narrative.
3. When statements cannot be extracted deterministically, the LLM transcribes inputs into
   `audit_financial_metrics_tool` (llm_fallback); any line item the parser did find fills the
   gaps the LLM left, and every override is logged.

The final artifact's numbers always come from the last deterministic tool result.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.filing.table_search import retrieve_10k_tables_tool, retrieve_multiyear_financial_series_tool
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.financial_auditor.context import (
    STATEMENT_TYPES,
    ExtractionResult,
    build_statement_grids,
    extract_audit_inputs,
    fetch_statement_tables,
    gap_fill_transcribed_inputs,
    resolve_audit_window,
)
from app.agents.specialized.financial_auditor.findings import SEVERITY_RANK, ForensicFinding
from app.agents.specialized.financial_auditor.modules.statement_integrity import (
    AuditDataQuality,
    InputCorrection,
    StatementIntegrity,
    apply_input_corrections,
    assess_statement_integrity,
    flag_data_quality,
    note_restatements,
    record_data_quality,
    summarize_for_judge,
)
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.tools.financial_math_tools import (
    NormalizationAdjustment,
    audit_financial_metrics_tool,
    run_audit_from_inputs,
)
from app.services.financial_tables import is_well_formed, render_clean_statement

logger = logging.getLogger("finance_agent.agents.financial_auditor")

AUDIT_TOOL_MARKERS = ("audit_financial_metrics", "refine_financial_audit")
ANALYTIC_KEYS = (
    "multi_year_history",
    "balance_sheet",
    "profitability_and_return_ratios",
    "solvency_and_liquidity_ratios",
    "earnings_quality",
    "working_capital",
    "capital_intensity",
    "cost_structure",
    "share_count_history",
    "forensic_findings",
)

REFINE_TOOL_DESCRIPTION = """Re-run the deterministic audit on the extracted 10-K inputs with cited corrections and/or normalization adjustments.

Use ONLY when (a) a baseline value contradicts the statement tables, or (b) reported earnings contain
non-recurring / non-core items that must be normalized. Never re-transcribe inputs; the extracted
inputs are already bound to this tool.

Args:
    normalization_adjustments: Items to strip from reported earnings. amount = POSITIVE magnitude in
        $ millions as disclosed. direction = "inflated_reported_earnings" (gains, credit sales, tax
        benefits, valuation-allowance releases -> normalized earnings fall) or
        "depressed_reported_earnings" (impairments, restructuring, litigation charges -> normalized
        earnings rise). affects = "operating_income" (pre-tax, inside EBIT) or "net_income"
        (below-the-line / tax items).
    corrections: Cited overrides of extracted values: target ("annual" | "balance_sheet" |
        "prior_balance_sheet"), fiscal_year, field (input field name), value, reason, source_chunk_id.
"""


@AgentRegistry.register("financial_auditor")
class FinancialAuditorAgent(StructuredAgent[FinancialAuditOutput]):
    """Deterministic-first auditor of multi-year 10-K financial statements."""

    prompt_name = "financial_auditor"
    tools = [retrieve_10k_tables_tool, retrieve_multiyear_financial_series_tool]
    output_schema = FinancialAuditOutput
    default_recursion_limit = 25

    def __init__(self, model_name: str = "openai:gpt-4o-mini", recursion_limit: Optional[int] = None):
        super().__init__(model_name=model_name, recursion_limit=recursion_limit)
        self._run_tools: Optional[List[Any]] = None

    def get_tools(self) -> List[Any]:
        return self._run_tools if self._run_tools is not None else self.tools + [audit_financial_metrics_tool]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def audit(
        self,
        ticker: str,
        fiscal_year: int,
        available_fiscal_years: Optional[List[int]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> FinancialAuditOutput:
        """Audits the 10-K statements for `ticker` / `fiscal_year` and returns a validated FinancialAuditOutput."""
        ticker = ticker.upper()
        target_years = resolve_audit_window(fiscal_year, available_fiscal_years)

        tables = fetch_statement_tables(ticker, fiscal_year, available_fiscal_years)
        extraction: Optional[ExtractionResult] = None
        if tables.get("income_statement"):
            try:
                extraction = extract_audit_inputs(build_statement_grids(tables), fiscal_year, target_years)
            except Exception as e:
                logger.warning(f"Statement extraction failed for {ticker} FY{fiscal_year}: {e}")
        integrity = assess_statement_integrity(extraction) if extraction is not None else None

        baseline_inputs: Optional[Dict[str, Any]] = None
        baseline: Optional[Dict[str, Any]] = None
        if extraction is not None and integrity.is_usable:
            baseline_inputs = extraction.to_tool_kwargs()
            try:
                baseline = run_audit_from_inputs(baseline_inputs)
            except Exception as e:
                extraction.warnings.append(f"Baseline audit rejected extracted inputs: {e}")
                logger.warning(f"Baseline audit failed for {ticker} FY{fiscal_year}: {e}")
                baseline_inputs = None

        mode = "deterministic" if baseline is not None else "llm_fallback"
        self._configure_tools(mode, baseline_inputs)
        logger.info(
            f"FinancialAuditor {ticker} FY{fiscal_year}: mode={mode}, "
            f"completeness={integrity.completeness_score if integrity else None}"
        )

        query = render_prompt(
            "financial_auditor", "prompt_financial_auditor_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            extraction_mode=mode,
            audit_years=extraction.years if mode == "deterministic" else target_years,
            available_years_str=", ".join(map(str, sorted(available_fiscal_years))) if available_fiscal_years else None,
            baseline_json=json.dumps(baseline, default=str) if baseline else None,
            extraction_json=json.dumps(summarize_for_judge(extraction, integrity), default=str) if extraction else None,
            clean_tables=extraction.clean_tables if extraction else {},
        )

        resolution = dict(
            extraction=extraction,
            integrity=integrity,
            extraction_mode=mode,
            baseline=baseline,
            baseline_inputs=baseline_inputs,
            prefetched_tables=tables,
        )
        fallback_defaults = {
            "auditor_summary": "Auditor narrative unavailable; refer to the deterministic analytics and findings.",
            "restatement_notes": [],
        }
        try:
            return self.execute_structured(
                query,
                ticker=ticker,
                fiscal_year=fiscal_year,
                fallback_defaults=fallback_defaults,
                callbacks=callbacks,
                **resolution,
            )
        except Exception as e:
            if baseline is None:
                raise
            # The LLM step only refines and narrates; the code-built baseline is a complete audit on its own.
            logger.warning(f"FinancialAuditor {ticker} FY{fiscal_year}: LLM step failed ({e!r}); using deterministic baseline.")
            fallback_defaults["auditor_summary"] = (
                "The auditor narrative step failed, so this audit is the deterministic baseline computed from the "
                "parsed statements without LLM refinement; refer to the tables and forensic findings."
            )
            empty: Dict[str, Any] = {"messages": []}
            output = self._extract_structured_output(
                empty, ticker=ticker, fiscal_year=fiscal_year, fallback_defaults=fallback_defaults, **resolution,
            )
            return self._post_process_output(output, empty, ticker=ticker, fiscal_year=fiscal_year, **resolution)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _configure_tools(self, mode: str, baseline_inputs: Optional[Dict[str, Any]]) -> None:
        if mode == "deterministic" and baseline_inputs is not None:
            self._run_tools = self.tools + [self._build_refine_tool(baseline_inputs)]
        else:
            self._run_tools = self.tools + [audit_financial_metrics_tool]
        self._cached_agent = None

    @staticmethod
    def _build_refine_tool(baseline_inputs: Dict[str, Any]) -> StructuredTool:
        def refine_financial_audit(
            normalization_adjustments: Optional[List[NormalizationAdjustment]] = None,
            corrections: Optional[List[InputCorrection]] = None,
        ) -> Dict[str, Any]:
            corrected = apply_input_corrections(baseline_inputs, corrections)
            audit = run_audit_from_inputs(corrected["inputs"], normalization_adjustments)
            audit["applied_corrections"] = corrected["applied"]
            audit["rejected_corrections"] = corrected["rejected"]
            return audit

        return StructuredTool.from_function(
            func=refine_financial_audit,
            name="refine_financial_audit_tool",
            description=REFINE_TOOL_DESCRIPTION,
        )

    # ------------------------------------------------------------------
    # Output lifecycle hooks
    # ------------------------------------------------------------------
    def _pre_validate_data(self, data: Dict[str, Any], result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._apply_resolution(data, result, **kwargs)

    def _post_process_output(
        self, output: FinancialAuditOutput, result: Dict[str, Any], **kwargs
    ) -> FinancialAuditOutput:
        merged = self._apply_resolution(output.model_dump(), result, **kwargs)
        return FinancialAuditOutput.model_validate(merged)

    def _apply_resolution(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any],
        extraction: Optional[ExtractionResult] = None,
        integrity: Optional[StatementIntegrity] = None,
        extraction_mode: str = "llm_fallback",
        baseline: Optional[Dict[str, Any]] = None,
        baseline_inputs: Optional[Dict[str, Any]] = None,
        prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Overlays the authoritative deterministic audit, data-quality record, tables and restatements."""
        messages = result.get("messages", [])
        log: List[str] = []
        stale = False

        tool_payload = self._last_audit_payload(messages)
        if extraction_mode == "deterministic":
            audit = tool_payload or baseline
            if tool_payload is None:
                log.append("No refinement requested by the auditor; deterministic baseline retained.")
            else:
                log.extend(self._describe_refinement(tool_payload))
        else:
            audit = tool_payload
            if tool_payload is not None and extraction is not None:
                args = self._last_audit_call_args(messages)
                if args:
                    gap_fill = gap_fill_transcribed_inputs(args, extraction)
                    log.extend(gap_fill["log"])
                    if gap_fill["changed"]:
                        audit = run_audit_from_inputs(gap_fill["inputs"], gap_fill["adjustments"])
                        stale = True

        if audit is not None:
            for key in ANALYTIC_KEYS:
                if key in audit:
                    data[key] = audit[key]

        dq = record_data_quality(extraction, integrity, extraction_mode, log, stale)
        data["data_quality"] = dq.model_dump()
        if "forensic_findings" in data:
            data["forensic_findings"] = self._merge_data_quality_findings(data["forensic_findings"], dq)

        tables = self._clean_tables(extraction, prefetched_tables, messages)
        for statement, field in (
            ("income_statement", "income_statement_markdown_table"),
            ("balance_sheet", "balance_sheet_markdown_table"),
            ("cash_flow", "cash_flow_markdown_table"),
        ):
            if tables.get(statement):
                data[field] = tables[statement]

        if not data.get("restatement_notes") and extraction is not None:
            data["restatement_notes"] = note_restatements(extraction, prefetched_tables)

        data["citations"] = self._merge_citations(data.get("citations") or [], prefetched_tables)
        return data

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _last_audit_payload(messages: List[Any]) -> Optional[Dict[str, Any]]:
        """Most recent successful audit / refine tool result."""
        for msg in reversed(messages):
            name = getattr(msg, "name", "") or ""
            if not any(marker in name for marker in AUDIT_TOOL_MARKERS):
                continue
            raw = getattr(msg, "content", "")
            payload = raw if isinstance(raw, dict) else None
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = None
            if isinstance(payload, dict) and "multi_year_history" in payload:
                return payload
        return None

    @staticmethod
    def _last_audit_call_args(messages: List[Any]) -> Optional[Dict[str, Any]]:
        for msg in reversed(messages):
            for call in reversed(getattr(msg, "tool_calls", None) or []):
                if "audit_financial_metrics" in (call.get("name") or ""):
                    return call.get("args") or None
        return None

    @staticmethod
    def _describe_refinement(payload: Dict[str, Any]) -> List[str]:
        log = []
        for c in payload.get("applied_corrections", []):
            log.append(
                f"Corrected {c['target']} FY{c['fiscal_year']}.{c['field']}: {c.get('previous_value')} -> "
                f"{c['value']} ({c['reason']}; source {c.get('source_chunk_id') or 'n/a'})."
            )
        for c in payload.get("rejected_corrections", []):
            log.append(f"Rejected correction {c.get('target')} FY{c.get('fiscal_year')}.{c.get('field')}: {c.get('rejection')}.")
        for a in payload.get("rejected_normalization_adjustments", []):
            log.append(f"Rejected normalization FY{a.get('fiscal_year')} '{a.get('label')}': {a.get('rejection')}.")
        for year in payload.get("earnings_quality", []):
            for adj in year.get("adjustments", []):
                effect = "added back" if adj["direction"] == "depressed_reported_earnings" else "removed"
                log.append(
                    f"Normalized FY{adj['fiscal_year']} {adj['affects']}: {effect} '{adj['label']}' "
                    f"({adj['amount']:,.1f}M, {adj['direction']}) — {adj.get('rationale') or 'no rationale given'}."
                )
        return log

    @staticmethod
    def _merge_data_quality_findings(findings: List[Any], dq: AuditDataQuality) -> List[Dict[str, Any]]:
        merged = [f.model_dump() if isinstance(f, ForensicFinding) else dict(f) for f in findings]
        merged = [f for f in merged if f.get("code") != "DATA_QUALITY"]
        merged.extend(f.model_dump() for f in flag_data_quality(dq))
        merged.sort(key=lambda f: (SEVERITY_RANK.get(f.get("severity"), 9), -(f.get("fiscal_year") or 0)))
        return merged

    @staticmethod
    def _clean_tables(
        extraction: Optional[ExtractionResult],
        prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]],
        messages: List[Any],
    ) -> Dict[str, str]:
        if extraction is not None and extraction.clean_tables:
            return extraction.clean_tables
        payloads: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STATEMENT_TYPES}
        for st, items in (prefetched_tables or {}).items():
            payloads.setdefault(st, []).extend(items)
        for msg in messages:
            name = getattr(msg, "name", "") or ""
            if "retrieve_10k_tables" not in name and "retrieve_multiyear" not in name:
                continue
            try:
                parsed = json.loads(getattr(msg, "content", "") or "[]")
            except Exception:
                continue
            for item in parsed if isinstance(parsed, list) else []:
                if isinstance(item, dict) and item.get("statement_type") in payloads and item.get("table_markdown"):
                    payloads[item["statement_type"]].append(item)
        tables: Dict[str, str] = {}
        for st, items in payloads.items():
            if not items:
                continue
            newest = sorted(items, key=lambda p: p.get("fiscal_year") or 0, reverse=True)
            grids = build_statement_grids({st: newest[:1]}).get(st) or []
            if grids and is_well_formed(grids[0], min_ratio=0.6):
                tables[st] = render_clean_statement(grids[0])
            else:
                tables[st] = newest[0]["table_markdown"]
        return tables

    @staticmethod
    def _merge_citations(
        citations: List[Dict[str, Any]],
        prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        prefetched = [
            {
                "chunk_id": p.get("chunk_id"),
                "item": p.get("item"),
                "breadcrumb": p.get("breadcrumb"),
                "sub_section": p.get("sub_section"),
                "fiscal_year": p.get("fiscal_year"),
                "statement_type": st,
            }
            for st, items in (prefetched_tables or {}).items()
            for p in items
        ]
        for cit in prefetched + list(citations):
            cid = cit.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            merged.append(cit)
        return merged


def audit_financial_statements(run: RunContext, upstream: Upstream) -> FinancialAuditOutput:
    """Pipeline entry: multi-year Item 8 audit anchored on the run's filing."""
    return FinancialAuditorAgent().audit(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        available_fiscal_years=list(run.available_fiscal_years),
        callbacks=run.callbacks,
    )


def summarize_audit(audit: Optional[FinancialAuditOutput]) -> StatusLine:
    if audit is None:
        return "Financial statement audit unavailable", None
    severe_cnt = sum(1 for f in audit.forensic_findings if f.severity in ("high", "medium"))
    return (
        f"Audited financial statements & ratios ({severe_cnt} high/medium forensic findings, "
        f"{audit.data_quality.extraction_mode} extraction)",
        None,
    )
