"""Assembles the Financial Auditor's output: the rows several modules feed, then the authoritative audit, data
quality, statement tables and citations."""

import json
from typing import Any, Dict, List, Optional

from app.agents.specialized.financial_auditor.context import STATEMENT_TYPES, ExtractionResult, build_statement_grids
from app.agents.specialized.financial_auditor.findings import SEVERITY_RANK, ForensicFinding
from app.agents.specialized.financial_auditor.modules.capital_allocation import CapitalReturns
from app.agents.specialized.financial_auditor.modules.cash_conversion import CashConversionYear
from app.agents.specialized.financial_auditor.modules.earnings_normalization import NormalizedYear
from app.agents.specialized.financial_auditor.modules.financial_history import YearFinancialsResult
from app.agents.specialized.financial_auditor.modules.statement_integrity import (
    AuditDataQuality,
    StatementIntegrity,
    flag_data_quality,
    note_restatements,
    record_data_quality,
)
from app.agents.specialized.financial_auditor.modules.tax_quality import LatestTaxRate, TaxYear
from app.agents.specialized.financial_auditor.state_financial_auditor import (
    ProfitabilityRatiosResult,
    YearEarningsQuality,
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


def assemble_profitability(
    latest_tax: LatestTaxRate,
    equity_status: str,
    returns: CapitalReturns,
    latest: YearFinancialsResult,
) -> ProfitabilityRatiosResult:
    """The latest year's tax rate (tax_quality), returns on capital (capital_allocation) and margins (history)."""
    return ProfitabilityRatiosResult(
        effective_tax_rate_pct=round(latest_tax.rate * 100.0, 2),
        tax_rate_source=latest_tax.source,
        equity_status=equity_status,
        nopat=returns.nopat,
        invested_capital=returns.invested_capital,
        capital_basis=returns.capital_basis,
        roic_pct=returns.roic_pct,
        roic_on_gross_capital_pct=returns.roic_on_gross_capital_pct,
        roe_pct=returns.roe_pct,
        latest_gross_margin_pct=latest.gross_margin_pct,
        latest_operating_margin_pct=latest.operating_margin_pct,
        latest_net_margin_pct=latest.net_margin_pct,
    )


def assemble_earnings_quality(
    cash_conversion: List[CashConversionYear],
    tax_years: List[TaxYear],
    normalized: List[NormalizedYear],
) -> List[YearEarningsQuality]:
    """One earnings-quality row per year from cash_conversion, tax_quality and earnings_normalization."""
    tax_by_year = {r.fiscal_year: r for r in tax_years}
    normalized_by_year = {r.fiscal_year: r for r in normalized}
    rows: List[YearEarningsQuality] = []
    for cc in cash_conversion:
        tx, nz = tax_by_year[cc.fiscal_year], normalized_by_year[cc.fiscal_year]
        rows.append(YearEarningsQuality(
            fiscal_year=cc.fiscal_year,
            accruals=cc.accruals,
            sloan_accruals_ratio_pct=cc.sloan_accruals_ratio_pct,
            stock_based_compensation=cc.stock_based_compensation,
            sbc_to_ocf_pct=cc.sbc_to_ocf_pct,
            sbc_to_operating_income_pct=cc.sbc_to_operating_income_pct,
            sbc_adjusted_fcf=cc.sbc_adjusted_fcf,
            non_operating_income=nz.non_operating_income,
            non_operating_to_operating_income_pct=nz.non_operating_to_operating_income_pct,
            effective_tax_rate_pct=tx.effective_tax_rate_pct,
            deferred_tax_to_net_income_pct=tx.deferred_tax_to_net_income_pct,
            adjustments=nz.adjustments,
            operating_income_adjustments_total=nz.operating_income_adjustments_total,
            net_income_adjustments_total=nz.net_income_adjustments_total,
            adjustments_pct_of_operating_income=nz.adjustments_pct_of_operating_income,
            normalized_operating_income=nz.normalized_operating_income,
            normalized_operating_margin_pct=nz.normalized_operating_margin_pct,
            normalized_net_income=nz.normalized_net_income,
            normalized_net_margin_pct=nz.normalized_net_margin_pct,
        ))
    return rows


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
