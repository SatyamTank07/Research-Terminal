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
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.financial_auditor.state_financial_auditor import (
    AuditDataQuality,
    FinancialAuditOutput,
)
from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    ForensicFinding,
    InputCorrection,
    NormalizationAdjustment,
    PriorBalanceSheetInput,
    apply_input_corrections,
    audit_financial_metrics_tool,
    run_audit_from_inputs,
)
from app.agents.tools.rag_table_tools import (
    retrieve_10k_tables,
    retrieve_10k_tables_tool,
    retrieve_multiyear_financial_series_tool,
)
from app.agents.tools.statement_extraction import (
    ExtractionResult,
    build_statement_grids,
    extract_audit_inputs,
)
from app.services.financial_tables import is_well_formed, render_clean_statement

logger = logging.getLogger("finance_agent.agents.financial_auditor")

STATEMENT_TYPES = ("income_statement", "balance_sheet", "cash_flow")
MAX_FILINGS = 3
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
GAP_FILL_DISAGREEMENT_PCT = 1.0
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}

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
        target_years = self._resolve_audit_window(fiscal_year, available_fiscal_years)

        tables = self._prefetch_statement_tables(ticker, fiscal_year, available_fiscal_years)
        extraction: Optional[ExtractionResult] = None
        if tables.get("income_statement"):
            try:
                extraction = extract_audit_inputs(build_statement_grids(tables), fiscal_year, target_years)
            except Exception as e:
                logger.warning(f"Statement extraction failed for {ticker} FY{fiscal_year}: {e}")

        baseline_inputs: Optional[Dict[str, Any]] = None
        baseline: Optional[Dict[str, Any]] = None
        if extraction is not None and extraction.is_usable:
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
            f"completeness={extraction.completeness_score if extraction else None}"
        )

        query = render_prompt(
            "prompt_financial_auditor_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            extraction_mode=mode,
            audit_years=extraction.years if mode == "deterministic" else target_years,
            available_years_str=", ".join(map(str, sorted(available_fiscal_years))) if available_fiscal_years else None,
            baseline_json=json.dumps(baseline, default=str) if baseline else None,
            extraction_json=json.dumps(extraction.summary_for_prompt(), default=str) if extraction else None,
            clean_tables=extraction.clean_tables if extraction else {},
        )

        return self.execute_structured(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults={
                "auditor_summary": "Auditor narrative unavailable; refer to the deterministic analytics and findings.",
                "restatement_notes": [],
            },
            callbacks=callbacks,
            extraction=extraction,
            extraction_mode=mode,
            baseline=baseline,
            baseline_inputs=baseline_inputs,
            prefetched_tables=tables,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_audit_window(fiscal_year: int, available_fiscal_years: Optional[List[int]]) -> List[int]:
        """3 contiguous years by default; up to 5 when a filing exists for every year in the span."""
        window = [fiscal_year - 2, fiscal_year - 1, fiscal_year]
        if available_fiscal_years:
            avail = {y for y in available_fiscal_years if y <= fiscal_year}
            for candidate_min in range(fiscal_year - 4, fiscal_year - 2):
                span = list(range(candidate_min, fiscal_year + 1))
                if all(y in avail for y in span):
                    return span
        return window

    def _prefetch_statement_tables(
        self,
        ticker: str,
        fiscal_year: int,
        available_fiscal_years: Optional[List[int]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieves the primary statements for the target filing and up to two prior filings (newest first)."""
        from app.database import SessionLocal

        prior = sorted((y for y in (available_fiscal_years or []) if y < fiscal_year), reverse=True)
        filing_years = [fiscal_year] + prior[: MAX_FILINGS - 1]
        tables: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STATEMENT_TYPES}
        db = None
        try:
            db = SessionLocal()
            for fy in filing_years:
                for st in STATEMENT_TYPES:
                    for res in retrieve_10k_tables(ticker, fy, statement_type=st, limit=1, db=db):
                        tables[st].append(res.model_dump())
        except Exception as e:
            logger.warning(f"Statement pre-fetch failed for {ticker} FY{fiscal_year}: {e}")
            return {}
        finally:
            if db is not None:
                db.close()
        return tables

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
                    gap_fill = self._gap_fill_llm_inputs(args, extraction)
                    log.extend(gap_fill["log"])
                    if gap_fill["changed"]:
                        audit = run_audit_from_inputs(gap_fill["inputs"], gap_fill["adjustments"])
                        stale = True

        if audit is not None:
            for key in ANALYTIC_KEYS:
                if key in audit:
                    data[key] = audit[key]

        dq = self._build_data_quality(extraction, extraction_mode, log, stale)
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
            data["restatement_notes"] = self._deterministic_restatement_notes(extraction, prefetched_tables)

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
    def _gap_fill_llm_inputs(args: Dict[str, Any], extraction: ExtractionResult) -> Dict[str, Any]:
        """Fills line items the LLM omitted (None, or 0 for securities/debt) with deterministically extracted values."""
        log: List[str] = []
        changed = False
        try:
            annual = [
                AnnualFinancialInput.model_validate(r).model_dump(exclude_none=True)
                for r in args.get("annual_financials", [])
            ]
            bs = BalanceSheetInput.model_validate(args["balance_sheet"]).model_dump(exclude_none=True)
            prior_raw = args.get("prior_balance_sheet")
            prior = PriorBalanceSheetInput.model_validate(prior_raw).model_dump(exclude_none=True) if prior_raw else None
        except Exception as e:
            return {"inputs": None, "adjustments": [], "log": [f"Gap-fill skipped: LLM inputs invalid ({e})."], "changed": False}

        extracted = {r["fiscal_year"]: r for r in extraction.to_annual_inputs_all_years()}
        zero_is_missing = {"marketable_securities", "short_term_debt", "long_term_debt"}

        def merge(record: Dict[str, Any], source: Dict[str, Any], label: str):
            nonlocal changed
            for fname, value in source.items():
                if fname == "fiscal_year" or value is None:
                    continue
                current = record.get(fname)
                if current is None or (fname in zero_is_missing and current == 0.0 and value > 0):
                    record[fname] = value
                    changed = True
                    log.append(f"Gap-filled {label}.{fname} = {value:,.2f} from parsed statements (LLM omitted it).")
                elif isinstance(current, (int, float)) and value:
                    diff = abs(current - value) / max(abs(value), 1e-9) * 100.0
                    if diff > GAP_FILL_DISAGREEMENT_PCT:
                        log.append(
                            f"Disagreement on {label}.{fname}: LLM {current:,.2f} vs parsed {value:,.2f} "
                            f"({diff:.1f}%); LLM value kept."
                        )

        for record in annual:
            merge(record, extracted.get(record["fiscal_year"], {}), f"FY{record['fiscal_year']}")
        merge(bs, extraction.to_balance_sheet_input(), f"FY{bs['fiscal_year']}.balance_sheet")
        extracted_prior = extraction.to_prior_balance_sheet_input()
        if prior is None and extracted_prior and extracted_prior["fiscal_year"] == bs["fiscal_year"] - 1:
            prior = extracted_prior
            changed = True
            log.append(f"Added FY{prior['fiscal_year']} prior balance sheet from parsed statements.")

        return {
            "inputs": {"annual_financials": annual, "balance_sheet": bs, "prior_balance_sheet": prior},
            "adjustments": args.get("normalization_adjustments") or [],
            "log": log,
            "changed": changed,
        }

    @staticmethod
    def _build_data_quality(
        extraction: Optional[ExtractionResult], mode: str, log: List[str], stale: bool
    ) -> AuditDataQuality:
        if extraction is None:
            return AuditDataQuality(
                extraction_mode=mode,
                reconciliation_log=log,
                warnings=["Primary statements could not be retrieved or parsed; inputs transcribed by the LLM."],
                summary_may_be_stale=stale,
            )
        return AuditDataQuality(
            extraction_mode=mode,
            completeness_score=extraction.completeness_score,
            missing_fields=extraction.missing_fields,
            identity_checks=extraction.identity_checks,
            cross_filing_differences=extraction.cross_filing_differences,
            reconciliation_log=log,
            field_provenance=extraction.field_provenance,
            warnings=extraction.warnings,
            summary_may_be_stale=stale,
        )

    @staticmethod
    def _merge_data_quality_findings(findings: List[Any], dq: AuditDataQuality) -> List[Dict[str, Any]]:
        merged = [f.model_dump() if isinstance(f, ForensicFinding) else dict(f) for f in findings]
        merged = [f for f in merged if f.get("code") != "DATA_QUALITY"]
        for check in dq.identity_checks:
            if check.get("status") == "failed":
                merged.append(ForensicFinding(
                    code="DATA_QUALITY", severity="high" if check.get("critical") else "medium",
                    metric="identity_check_difference_pct", value=check.get("difference_pct"),
                    threshold=None, message=f"DATA QUALITY: Identity check failed — {check['name']} "
                    f"({check['lhs']:,.1f} vs {check['rhs']:,.1f}, {check['difference_pct']:.2f}% apart).",
                ).model_dump())
        if dq.extraction_mode == "llm_fallback":
            merged.append(ForensicFinding(
                code="DATA_QUALITY", severity="info", metric="extraction_mode", threshold=None,
                message="DATA QUALITY: Statements were not machine-parsed; inputs were transcribed by the LLM "
                        "and gap-filled where parsed values existed. Verify key figures against the filing.",
            ).model_dump())
        merged.sort(key=lambda f: (_SEVERITY_RANK.get(f.get("severity"), 9), -(f.get("fiscal_year") or 0)))
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
    def _deterministic_restatement_notes(
        extraction: ExtractionResult,
        prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]],
    ) -> List[str]:
        notes = [
            f"FY{d['fiscal_year']} {d['field'].replace('_', ' ')} ({d['statement']}): "
            f"{d['older_value']:,.1f} in the FY{d['older_filing_year']} 10-K vs {d['latest_value']:,.1f} in the "
            f"FY{d['latest_filing_year']} 10-K ({d['difference_pct']:.2f}% difference); latest filing figure used."
            for d in extraction.cross_filing_differences
        ]
        if notes:
            return notes
        filings = sorted(
            {p.get("fiscal_year") for items in (prefetched_tables or {}).values() for p in items if p.get("fiscal_year")},
            reverse=True,
        )
        if len(filings) > 1:
            return [
                "No retrospective restatements identified: overlapping years agree across the "
                + ", ".join(f"FY{y}" for y in filings) + " 10-K filings for all extracted line items."
            ]
        return ["Single filing available; cross-filing restatement review not possible."]

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
