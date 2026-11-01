"""Builds the Risk Analyst's output from the modules' results: data-quality counts and notes in the order the report
has always shown them, the citations behind the verified evidence, and the two markdown tables."""

from typing import Any, Dict, List, Optional, Sequence

from app.agents.specialized.risk_analyst.modules.disclosure_changes import RiskDisclosureDiff
from app.agents.specialized.risk_analyst.modules.quantified_exposures import QuantifiedCoverage
from app.agents.specialized.risk_analyst.modules.risk_inventory import InventoryResult, RiskItem
from app.agents.specialized.risk_analyst.modules.risk_profile import (
    PrimaryThreat,
    RankedRisks,
    derive_overall_profile,
    render_risk_matrix_markdown,
    table_cell,
)
from app.agents.specialized.risk_analyst.state_risk_analyst import RiskAuditOutput, RiskDataQuality

Pool = Dict[str, Dict[str, Any]]

NOT_PROVIDED_MATRIX = (
    "_The filing provides no Item 1A risk factors (not applicable / smaller reporting company); no risks were rated._"
)
_CITATION_KEYS = ("chunk_id", "ticker", "fiscal_year", "item", "breadcrumb", "sub_section")


def assemble_risk_audit(
    output: RiskAuditOutput,
    inventory: InventoryResult,
    profile: RankedRisks,
    coverage: QuantifiedCoverage,
    threat: PrimaryThreat,
    pool: Pool,
    diff: Optional[RiskDisclosureDiff],
    dq: RiskDataQuality,
) -> RiskAuditOutput:
    """Fills `output` with the ranked risks, the primary threat, the profile, citations, tables and data quality."""
    dq.quotes_rejected += inventory.quotes_rejected
    dq.risks_dropped_unverified += inventory.dropped_unverified
    dq.risks_merged = len(inventory.absorbed)
    dq.notes.extend(profile.notes)
    dq.quantified_uncovered = list(coverage.listed)
    if coverage.note:
        dq.notes.append(coverage.note)
    if threat.note:
        dq.notes.append(threat.note)

    ranked = profile.risks
    output.identified_risks = ranked
    output.primary_threat_risk_id = threat.risk_id
    output.primary_existential_threat = threat.statement
    output.overall_risk_profile = derive_overall_profile([r.severity for r in ranked])
    output.citations = build_citations(ranked, pool)
    output.risk_matrix_markdown = render_risk_matrix_markdown(ranked)
    output.evidence_table_markdown = render_risk_evidence_table(output.citations)
    output.disclosure_changes = diff
    output.data_quality = dq
    return output


def assemble_not_provided(
    ticker: str, fiscal_year: int, chunks: Sequence[Any], located_by: str
) -> RiskAuditOutput:
    """Item 1A exists but discloses no risk factors (e.g. smaller reporting company): nothing is rated."""
    statement = " ".join(c.content for c in chunks).strip()
    citations = [{k: v for k, v in c.model_dump().items() if k in _CITATION_KEYS} for c in chunks]
    note = (
        f"The FY{fiscal_year} 10-K provides no Item 1A risk factors"
        + (f' (filing states: "{statement[:200]}")' if statement else "")
        + "; no risks were rated."
    )
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
        risk_matrix_markdown=NOT_PROVIDED_MATRIX,
        evidence_table_markdown=render_risk_evidence_table(citations),
        citations=citations,
    )


def build_citations(risks: Sequence[RiskItem], pool: Pool) -> List[Dict[str, Any]]:
    """One citation per chunk behind the verified evidence and mitigations, in risk order, with its filing location."""
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
            for key in _CITATION_KEYS[1:]:
                if chunk.get(key) is not None:
                    citation[key] = chunk[key]
            citations.append(citation)
    return citations


def render_risk_evidence_table(citations: Sequence[Dict[str, Any]]) -> str:
    if not citations:
        return "_No verified filing evidence._"
    lines = ["| Chunk ID | Section | Breadcrumb |", "| :--- | :--- | :--- |"]
    for c in citations:
        lines.append(f"| `{table_cell(c.get('chunk_id'))}` | {table_cell(c.get('item'))} | {table_cell(c.get('breadcrumb'))} |")
    return "\n".join(lines)
