"""The verified likelihood x impact risk matrix as a markdown table."""

from typing import Any, Sequence


def table_cell(text: Any) -> str:
    """A markdown table cell: an em dash when empty, no pipes or line breaks."""
    return str(text if text not in (None, "") else "—").replace("|", "/").replace("\n", " ")


def render_risk_matrix_markdown(risks: Sequence[Any]) -> str:
    """Renders the verified likelihood x impact risk matrix (accepts RiskItem models or dicts)."""
    if not risks:
        return "_No risks survived evidence verification against the filing._"
    get = (lambda r, k: r.get(k)) if isinstance(risks[0], dict) else getattr
    lines = [
        "| # | Risk | Category | Likelihood | Impact | Severity | Model line(s) hit | vs prior year |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |",
    ]
    for r in risks:
        transmission = get(r, "financial_transmission") or []
        lines.append(
            f"| {table_cell(get(r, 'risk_id'))} | {table_cell(get(r, 'risk_title'))} | {table_cell(get(r, 'risk_category'))} | "
            f"{table_cell(get(r, 'likelihood'))} | {table_cell(get(r, 'impact'))} | **{table_cell(get(r, 'severity'))}** | "
            f"{table_cell(', '.join(transmission))} | {table_cell(get(r, 'disclosure_change'))} |"
        )
    return "\n".join(lines)
