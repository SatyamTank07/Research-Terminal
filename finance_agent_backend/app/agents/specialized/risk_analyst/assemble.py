"""Builds the Risk Analyst's output pieces from the modules' results."""

from typing import Any, Dict, Sequence

from app.agents.specialized.risk_analyst.modules.risk_profile import table_cell


def render_risk_evidence_table(citations: Sequence[Dict[str, Any]]) -> str:
    if not citations:
        return "_No verified filing evidence._"
    lines = ["| Chunk ID | Section | Breadcrumb |", "| :--- | :--- | :--- |"]
    for c in citations:
        lines.append(f"| `{table_cell(c.get('chunk_id'))}` | {table_cell(c.get('item'))} | {table_cell(c.get('breadcrumb'))} |")
    return "\n".join(lines)
