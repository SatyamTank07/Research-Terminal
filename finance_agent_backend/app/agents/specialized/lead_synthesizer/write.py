"""The agent's write step: one LLM call writes the route's prose fields; code screens them and backs each one up.

Prose is screened in code: sentences citing figures absent from the verified context are removed, invented dates
and unsupported year-over-year risk claims are stripped, and every field has a deterministic fallback, so a failed
LLM call still yields a complete report.
"""

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.agents.shared.figures import collect_known_figures, strip_unverified_sentences
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.specialized.risk_analyst import RiskAuditOutput
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput

logger = logging.getLogger("finance_agent.agents.lead_synthesizer")

SynthesisProvenance = Literal["llm_structured", "llm_json_fallback", "template_default"]

_DATE_CLAIM = re.compile(
    r"\b(?:analysis|report|publication)\s+date\b|\bas of (?:today|the date of this report)\b", re.IGNORECASE
)
_YOY_CLAIM = re.compile(
    r"year[- ]over[- ]year|\b(?:prior|previous|last)[- ](?:fiscal[- ])?year\b|"
    r"\bcompared\s+(?:with|to)\s+(?:the\s+)?(?:prior|previous|last)\b|\bnewly (?:added|disclosed)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\"(\[])")


class ResearchSynthesisPayload(BaseModel):
    """Prose fields written by the LLM; which ones are requested depends on the route."""

    executive_summary: Optional[str] = Field(None, description="Executive summary (bullets for the audit route)")
    bottom_line: Optional[str] = Field(None, description="1-2 sentence stance and what would change it")
    pillar_1_business_moat: Optional[str] = Field(None, description="Pillar 1: business architecture and moat")
    pillar_2_financial_durability: Optional[str] = Field(None, description="Pillar 2: earnings quality and balance sheet")
    pillar_3_valuation_asymmetry: Optional[str] = Field(None, description="Pillar 3: value vs price and margin of safety")
    business_commentary: Optional[str] = Field(None, description="Moat and segment interpretation")
    earnings_quality_commentary: Optional[str] = Field(None, description="Reported vs normalized earnings interpretation")
    cash_flow_balance_sheet_commentary: Optional[str] = Field(None, description="FCF drivers, dilution, liquidity")
    forecast_commentary: Optional[str] = Field(None, description="What drives the UFCF path")
    valuation_commentary: Optional[str] = Field(None, description="Valuation interpretation incl. every warning")
    risk_commentary: Optional[str] = Field(None, description="Risk summary")
    bear_case_commentary: Optional[str] = Field(None, description="Bear-case linkage of the primary threat")
    disclosure_change_commentary: Optional[str] = Field(None, description="Year-over-year Item 1A changes")


PROSE_FIELDS = tuple(ResearchSynthesisPayload.model_fields)
_RISK_PROSE = ("risk_commentary", "bear_case_commentary", "disclosure_change_commentary", "executive_summary", "bottom_line")


def drop_sentences(text: str, pattern: re.Pattern) -> str:
    """Removes sentences matching `pattern`, line by line (bullet prefixes are kept)."""
    lines = []
    for line in text.split("\n"):
        prefix = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+|\s*)", line).group(1)
        kept = " ".join(s for s in _SENTENCE_BOUNDARY.split(line[len(prefix):]) if not pattern.search(s)).strip()
        if kept:
            lines.append(prefix + kept)
    return "\n".join(lines).strip()


def strip_layout(text: str) -> str:
    """Prose must not carry its own headings, tables or code fences (the layout owns them)."""
    lines = [
        line for line in text.replace("```", "").split("\n")
        if not line.lstrip().startswith(("#", "|"))
    ]
    return "\n".join(lines).strip()


def render_write_prompts(
    ticker: str,
    company_name: str,
    fiscal_year: int,
    available_fiscal_years: Optional[List[int]],
    query_type: str,
    user_query: Optional[str],
    year_substituted: bool,
    pipeline_notes: List[str],
    artifacts: Dict[str, Any],
) -> Tuple[str, str]:
    """(system context, instruction): the verified upstream context and the route's prose mandate."""
    context = render_prompt(
        "lead_synthesizer", "prompt_lead_synthesizer.j2",
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        available_fiscal_years=available_fiscal_years,
        query_type=query_type,
        user_query=user_query,
        year_substituted=year_substituted,
        **artifacts,
    )
    instruction = render_prompt(
        "lead_synthesizer", "write.j2",
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        query_type=query_type,
        user_query=user_query,
        pipeline_notes=pipeline_notes,
    )
    return context, instruction


def screen_prose(raw: Dict[str, Any], context: str, risk_audit: Optional[RiskAuditOutput]) -> Tuple[Dict[str, Optional[str]], int]:
    """Keeps only verified prose: no layout, no invented dates, no untraceable figures, no baseless YoY claims."""
    known = collect_known_figures(context)
    no_risk_baseline = risk_audit is not None and risk_audit.disclosure_changes is None
    prose: Dict[str, Optional[str]] = {}
    removed_total = 0
    for key in PROSE_FIELDS:
        text = raw.get(key)
        if not isinstance(text, str) or not text.strip():
            prose[key] = None
            continue
        text = drop_sentences(strip_layout(text), _DATE_CLAIM)
        if no_risk_baseline and key in _RISK_PROSE:
            text = drop_sentences(text, _YOY_CLAIM)
        text, removed = strip_unverified_sentences(text, known)
        removed_total += removed
        prose[key] = text or None
    if removed_total:
        logger.warning(f"Synthesis: removed {removed_total} sentence(s) citing figures absent from the context.")
    return prose, removed_total


def write_fallback_prose(
    ticker: str,
    company_name: str,
    fiscal_year: int,
    business_moat: Optional[BusinessMoatOutput],
    financial_audit: Optional[FinancialAuditOutput],
    forecast: Optional[ForecastOutput],
    dcf_valuation: Optional[DCFValuationOutput],
    risk_audit: Optional[RiskAuditOutput],
) -> Dict[str, str]:
    """Deterministic commentary per field, used when the LLM leaves a field empty or fails."""
    out: Dict[str, str] = {}
    summary: List[str] = [f"{company_name} ({ticker}), FY{fiscal_year} Form 10-K."]

    if business_moat is not None:
        out["business_commentary"] = business_moat.moat_rationale
        out["pillar_1_business_moat"] = (
            f"Primary moat: {business_moat.economic_moat_type} ({business_moat.moat_durability} durability, "
            f"{business_moat.moat_trajectory} trajectory). {business_moat.moat_rationale}"
        )
        summary.append(f"The business strategist classifies the primary moat as {business_moat.economic_moat_type}.")
    else:
        out["pillar_1_business_moat"] = "The business and moat analysis is unavailable for this report."

    if financial_audit is not None:
        latest = financial_audit.multi_year_history[-1]
        serious = [f for f in financial_audit.forensic_findings if f.severity in ("high", "medium")]
        out["earnings_quality_commentary"] = financial_audit.auditor_summary
        out["pillar_2_financial_durability"] = (
            f"FY{latest.fiscal_year} revenue was ${latest.revenue:,.1f}M with an operating margin of "
            f"{latest.operating_margin_pct:.2f}% and free cash flow of ${latest.free_cash_flow:,.1f}M; "
            f"net debt is ${financial_audit.balance_sheet.net_debt:,.1f}M (negative = net cash). "
            f"{len(serious)} high/medium forensic finding(s) were flagged."
        ) if latest.operating_margin_pct is not None else financial_audit.auditor_summary
        out["cash_flow_balance_sheet_commentary"] = (
            "See Tables D and E for the FCF bridge, capital intensity, dilution, liquidity and returns."
        )
        summary.append(f"The audit flagged {len(serious)} high/medium forensic finding(s).")
    else:
        out["pillar_2_financial_durability"] = "The financial statement audit is unavailable for this report."

    if forecast is not None:
        out["forecast_commentary"] = (
            f"The base case projects a {forecast.revenue_cagr_pct:.1f}% revenue CAGR over "
            f"{forecast.forecast_horizon_years} years and cumulative UFCF of ${forecast.cumulative_fcf:,.1f}M "
            f"({forecast.guidance_source.replace('_', ' ')})."
        )

    if dcf_valuation is not None:
        v = dcf_valuation
        out["valuation_commentary"] = v.valuation_summary
        if v.current_share_price is not None:
            valuation_line = (
                f"The base-case DCF fair value is ${v.implied_fair_value_per_share:,.2f} per share against a "
                f"${v.current_share_price:,.2f} share price ({v.upside_downside_pct:+.1f}%, {v.valuation_stance})."
            )
        else:
            valuation_line = (
                f"The base-case DCF fair value is ${v.implied_fair_value_per_share:,.2f} per share; no market price "
                "was available, so no stance is stated."
            )
        out["pillar_3_valuation_asymmetry"] = valuation_line
        out["bottom_line"] = valuation_line
        summary.append(valuation_line)
    else:
        out["pillar_3_valuation_asymmetry"] = "The DCF valuation is unavailable for this report."

    if risk_audit is not None:
        threat = risk_audit.primary_existential_threat or "No primary existential threat was identified."
        out["risk_commentary"] = (
            f"- Primary existential threat: {threat}\n"
            f"- Overall risk profile: {risk_audit.overall_risk_profile}.\n"
            + (
                f"- Disclosure baseline: compared against the FY{risk_audit.disclosure_changes.prior_fiscal_year} Item 1A."
                if risk_audit.disclosure_changes else "- No prior-year Item 1A baseline was available."
            )
        )
        primary = next((r for r in risk_audit.identified_risks if r.risk_id == risk_audit.primary_threat_risk_id), None)
        out["bear_case_commentary"] = (
            f"A bear case should stress the model lines the primary threat ({primary.risk_id}) hits: "
            f"{', '.join(primary.financial_transmission)}."
            if primary is not None and primary.financial_transmission else threat
        )
        out["disclosure_change_commentary"] = ""
        summary.append(f"The overall risk profile is {risk_audit.overall_risk_profile}.")

    out["executive_summary"] = " ".join(summary)
    return out


def fill_empty_prose(prose: Dict[str, Optional[str]], fallback: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Every field the LLM left empty (or that the screen emptied) takes its deterministic fallback."""
    return {key: prose.get(key) or fallback.get(key) for key in PROSE_FIELDS}
