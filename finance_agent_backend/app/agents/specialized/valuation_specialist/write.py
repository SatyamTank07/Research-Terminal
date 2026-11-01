"""The agent's write step: one LLM call turns the code-built valuation into commentary; a template backs it up."""

import logging
import re
from typing import Callable, Literal, Tuple

from app.agents.shared.prompts import render_prompt
from app.agents.specialized.valuation_specialist.state_valuation_specialist import ValuationBuild

logger = logging.getLogger("finance_agent.agents.valuation_specialist")

MIN_COMMENTARY_CHARS = 200
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
CommentarySource = Literal["llm", "template"]


def trim_to_complete_sentence(text: str) -> str:
    """Drops a trailing fragment left when the completion hit its token cap mid-sentence."""
    text = text.strip()
    ends = list(_SENTENCE_END.finditer(text))
    if not ends or ends[-1].end() == len(text):
        return text
    return text[: ends[-1].end()].rstrip()


def write_valuation_summary(build: ValuationBuild, complete: Callable[[str], str]) -> Tuple[str, CommentarySource]:
    """Commentary over the final figures. A failed, empty or too-short completion falls back to the template."""
    summary, source = None, "template"
    try:
        summary = complete(render_prompt("valuation_specialist", "write.j2", v=build))
        summary = trim_to_complete_sentence(summary or "")
        if len(summary) >= MIN_COMMENTARY_CHARS:
            source = "llm"
        else:
            logger.warning(f"Valuation commentary for {build.ticker} too short ({len(summary)} chars); using template.")
    except Exception as e:
        logger.warning(f"Valuation commentary LLM failed for {build.ticker}: {e}; using template.")
    if source == "template":
        summary = summarize_from_template(build)
    return summary, source


def summarize_from_template(v: ValuationBuild) -> str:
    """Deterministic commentary used when the LLM call fails or returns too little."""
    w = v.wacc_audit
    paragraphs = [
        f"Valuation bridge: the base-case DCF implies an enterprise value of ${v.enterprise_value:,.0f}M, "
        f"of which {v.terminal_value_pct_of_ev:.1f}% comes from the terminal value "
        f"({v.terminal_method.replace('_', ' ')}, {v.terminal_growth_rate * 100:.1f}% perpetual growth). "
        f"After net debt of ${v.net_debt:,.0f}M and noncontrolling interests of ${v.noncontrolling_interests:,.0f}M, "
        f"equity value is ${v.equity_value:,.0f}M, or ${v.implied_fair_value_per_share:.2f} per share.",
        f"Hurdle rate: the applied discount rate is {w.wacc_pct:.2f}% ({w.wacc_source.replace('_', ' ')}); "
        f"CAPM cost of equity is {w.cost_of_equity_pct:.2f}% (beta {w.beta:.2f}, Rf {w.risk_free_rate_pct:.2f}%, "
        f"ERP {w.equity_risk_premium_pct:.2f}%) and after-tax cost of debt is {w.cost_of_debt_after_tax_pct:.2f}% "
        f"at a {w.debt_weight_pct:.1f}% debt weight.",
    ]
    grids = "WACC x growth and WACC x RONIC grids" if v.ronic_sensitivity_matrix_markdown else "WACC x growth grid"
    if v.bear_fair_value is not None and v.bull_fair_value is not None:
        paragraphs.append(
            f"Range: bear ${v.bear_fair_value:.2f}, base ${v.implied_fair_value_per_share:.2f}, "
            f"bull ${v.bull_fair_value:.2f}; probability-weighted ${v.probability_weighted_fair_value:.2f}. "
            f"The {grids} bracket the base case."
        )
    if v.current_share_price is not None:
        paragraphs.append(
            f"Margin of safety: against a ${v.current_share_price:.2f} share price the base case implies "
            f"{v.upside_downside_pct:+.1f}% ({v.valuation_stance}). Reverse DCF: {v.reverse_dcf_summary}"
        )
    else:
        paragraphs.append("Margin of safety: no market price was available, so no upside or stance is stated.")
    if v.data_quality.warnings:
        paragraphs.append("Caveats: " + " ".join(v.data_quality.warnings))
    return "\n\n".join(paragraphs)
