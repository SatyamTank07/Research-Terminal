"""The bull / base / bear scenario table, with the probability-weighted row."""

from typing import List, Optional

from app.agents.specialized.valuation_specialist.modules.scenario_values.schema import ScenarioValuation


def render_scenario_table(
    scenarios: List[ScenarioValuation], weighted: Optional[float], weighted_upside_pct: Optional[float]
) -> str:
    lines = [
        "| Scenario | Probability | Revenue CAGR | Cumulative UFCF | Fair Value / Share | vs. Price |",
        "|:---|:---:|:---:|---:|---:|:---:|",
    ]
    for s in scenarios:
        cagr = f"{s.revenue_cagr_pct:.1f}%" if s.revenue_cagr_pct is not None else "n/a"
        vs = f"{s.upside_downside_pct:+.1f}%" if s.upside_downside_pct is not None else "n/a"
        lines.append(
            f"| {s.name.title()} | {s.probability:.0%} | {cagr} | ${s.cumulative_fcf:,.0f}M | "
            f"${s.fair_value_per_share:.2f} | {vs} |"
        )
    if weighted is not None:
        vs = weighted_upside_pct
        lines.append(
            f"| **Probability-weighted** | 100% | | | **${weighted:.2f}** | "
            f"{f'{vs:+.1f}%' if vs is not None else 'n/a'} |"
        )
    return "\n".join(lines)
