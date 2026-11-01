"""Assembles the Financial Auditor's output: the authoritative audit numbers, data quality, statement tables, citations."""

import json
from typing import Any, Dict, List, Optional

from app.agents.specialized.financial_auditor.context import STATEMENT_TYPES, ExtractionResult, build_statement_grids
from app.agents.specialized.financial_auditor.findings import SEVERITY_RANK, ForensicFinding
from app.agents.specialized.financial_auditor.modules.statement_integrity import (
    AuditDataQuality,
    StatementIntegrity,
    flag_data_quality,
    note_restatements,
    record_data_quality,
)
from app.services.financial_tables import is_well_formed, render_clean_statement

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


def assemble_audit(
    data: Dict[str, Any],
    *,
    audit: Optional[Dict[str, Any]],
    log: List[str],
    stale: bool,
    extraction: Optional[ExtractionResult],
    integrity: Optional[StatementIntegrity],
    extraction_mode: str,
    prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]],
    messages: List[Any],
) -> Dict[str, Any]:
    """Overlays the authoritative deterministic audit, data-quality record, tables and restatements."""
    if audit is not None:
        for key in ANALYTIC_KEYS:
            if key in audit:
                data[key] = audit[key]

    dq = record_data_quality(extraction, integrity, extraction_mode, log, stale)
    data["data_quality"] = dq.model_dump()
    if "forensic_findings" in data:
        data["forensic_findings"] = merge_data_quality_findings(data["forensic_findings"], dq)

    tables = clean_tables(extraction, prefetched_tables, messages)
    for statement, field in (
        ("income_statement", "income_statement_markdown_table"),
        ("balance_sheet", "balance_sheet_markdown_table"),
        ("cash_flow", "cash_flow_markdown_table"),
    ):
        if tables.get(statement):
            data[field] = tables[statement]

    if not data.get("restatement_notes") and extraction is not None:
        data["restatement_notes"] = note_restatements(extraction, prefetched_tables)

    data["citations"] = merge_citations(data.get("citations") or [], prefetched_tables)
    return data


def merge_data_quality_findings(findings: List[Any], dq: AuditDataQuality) -> List[Dict[str, Any]]:
    """Replaces the DATA_QUALITY findings with this run's; ranked findings keep their order within a tie."""
    merged = [f.model_dump() if isinstance(f, ForensicFinding) else dict(f) for f in findings]
    merged = [f for f in merged if f.get("code") != "DATA_QUALITY"]
    merged.extend(f.model_dump() for f in flag_data_quality(dq))
    merged.sort(key=lambda f: (SEVERITY_RANK.get(f.get("severity"), 9), -(f.get("fiscal_year") or 0)))
    return merged


def clean_tables(
    extraction: Optional[ExtractionResult],
    prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]],
    messages: List[Any],
) -> Dict[str, str]:
    """Clean markdown statement tables from the parsed grids, else from the newest retrieved table."""
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


def merge_citations(
    citations: List[Dict[str, Any]],
    prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Pre-fetched statement chunks first, then the LLM's citations, one entry per chunk."""
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
