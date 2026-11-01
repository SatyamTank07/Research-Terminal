"""Lead Synthesizer & Institutional Report Publisher Agent.

Code-assembled report: the route-specific skeleton and every table (dashboard, audited statements and
Tables A–F, forecast schedule, WACC, equity bridge, scenarios, sensitivity, risk matrix and evidence) are
rendered in code from the specialist artifacts (`report_layout.j2`). One structured LLM call writes only
the analytical prose between them. Prose is screened in code: sentences citing figures absent from the
verified context are removed, invented dates and unsupported year-over-year risk claims are stripped,
and every field has a deterministic fallback, so a failed LLM call still yields a complete report.
Every filing-backed table row, quote and prose figure carries an inline [[n]](#cite-n) marker built in code
(report_citations.py); the numbered sources ship as Final10KResearchReport.sources.
"""

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.base import AgentOutput, BaseAgent, StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.citations import CitationRegistry, SourceRef, strip_markers
from app.agents.shared.figures import (
    annotate_sentences,
    build_figure_index,
    collect_known_figures,
    strip_unverified_sentences,
)
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.specialized.lead_synthesizer.state_lead_synthesizer import (
    Final10KResearchReport,
    ThreePillarThesis,
)
from app.agents.specialized.risk_analyst import RiskAuditOutput
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput
from app.agents.specialized.lead_synthesizer.report_citations import (
    CITATIONS_FAILED_NOTE,
    EVIDENCE_NOTE,
    NO_CITATIONS_NOTE,
    NullCiter,
    ReportCiter,
)

logger = logging.getLogger("finance_agent.agents.lead_synthesizer")

SYNTHESIS_MAX_TOKENS = 4000
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


def _drop_sentences(text: str, pattern: re.Pattern) -> str:
    """Removes sentences matching `pattern`, line by line (bullet prefixes are kept)."""
    lines = []
    for line in text.split("\n"):
        prefix = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+|\s*)", line).group(1)
        kept = " ".join(s for s in _SENTENCE_BOUNDARY.split(line[len(prefix):]) if not pattern.search(s)).strip()
        if kept:
            lines.append(prefix + kept)
    return "\n".join(lines).strip()


def _strip_layout(text: str) -> str:
    """Prose must not carry its own headings, tables or code fences (the layout owns them)."""
    lines = [
        line for line in text.replace("```", "").split("\n")
        if not line.lstrip().startswith(("#", "|"))
    ]
    return "\n".join(lines).strip()


@AgentRegistry.register("lead_synthesizer")
class LeadSynthesizerAgent(BaseAgent):
    """Institutional report publisher: code-built report skeleton and tables, LLM-written verified prose."""

    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        self.model_name = model_name
        self._cached_llm = None

    def _get_llm(self):
        """Cache-first LLM resolver to eliminate redundant disk I/O from load_dotenv."""
        if self._cached_llm is not None:
            return self._cached_llm
        load_dotenv(override=True)
        self._cached_llm = ChatOpenAI(
            model=self.model_name.replace("openai:", ""),
            temperature=0,
            max_retries=5,
            max_tokens=SYNTHESIS_MAX_TOKENS,
        )
        return self._cached_llm

    def synthesize(
        self,
        ticker: str,
        company_name: str,
        fiscal_year: int,
        business_moat: Optional[BusinessMoatOutput] = None,
        financial_audit: Optional[FinancialAuditOutput] = None,
        forecast: Optional[ForecastOutput] = None,
        dcf_valuation: Optional[DCFValuationOutput] = None,
        risk_audit: Optional[RiskAuditOutput] = None,
        year_substituted: bool = False,
        available_fiscal_years: Optional[List[int]] = None,
        user_query: Optional[str] = None,
        query_type: str = "full_10k_report",
        pipeline_notes: Optional[List[str]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Final10KResearchReport:
        """Synthesizes all structured agent payloads into a complete Final10KResearchReport."""
        ticker = ticker.upper()
        notes = list(dict.fromkeys(pipeline_notes or []))
        artifacts = dict(
            business_moat=business_moat,
            financial_audit=financial_audit,
            forecast=forecast,
            dcf_valuation=dcf_valuation,
            risk_audit=risk_audit,
        )
        query_type = self._effective_route(query_type, artifacts)

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
            "lead_synthesizer", "prompt_synthesizer_instruction.j2",
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
            query_type=query_type,
            user_query=user_query,
            pipeline_notes=notes,
        )

        raw, provenance = self._invoke_llm(context, instruction, callbacks)
        prose, removed = self._screen_prose(raw, context, risk_audit)
        fallback = self._fallback_prose(ticker, company_name, fiscal_year, **artifacts)
        for key in PROSE_FIELDS:
            if not prose.get(key):
                prose[key] = fallback.get(key)

        citer, cited_prose = self._cite_prose(prose, fiscal_year, artifacts, notes)
        full_report_md = render_prompt(
            "lead_synthesizer", "report_layout.j2",
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
            query_type=query_type,
            year_substituted=year_substituted,
            p=cited_prose,
            notes=notes,
            provenance=provenance,
            removed_sentences=removed,
            cite=citer,
            **artifacts,
        )
        full_report_md = re.sub(r"\n{3,}", "\n\n", full_report_md).strip() + "\n"
        full_report_md, sources = self._finalize_citations(citer, full_report_md)

        val = dcf_valuation
        return Final10KResearchReport(
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
            implied_fair_value_per_share=val.implied_fair_value_per_share if val else None,
            current_share_price=val.current_share_price if val else None,
            upside_downside_pct=val.upside_downside_pct if val else None,
            valuation_stance=val.valuation_stance if val else None,
            three_pillar_thesis=ThreePillarThesis(
                pillar_1_business_moat=prose["pillar_1_business_moat"],
                pillar_2_financial_durability=prose["pillar_2_financial_durability"],
                pillar_3_valuation_asymmetry=prose["pillar_3_valuation_asymmetry"],
            ),
            executive_summary=prose["executive_summary"],
            synthesis_provenance=provenance,
            year_substituted=year_substituted,
            full_markdown_report=full_report_md,
            sources=sources,
        )

    # ------------------------------------------------------------------
    # LLM call and prose screening
    # ------------------------------------------------------------------
    @staticmethod
    def _effective_route(query_type: str, artifacts: Dict[str, Any]) -> str:
        """Falls back to the full layout when the artifact a single-topic route needs is missing."""
        required = {
            "financial_audit_only": "financial_audit",
            "business_moat_only": "business_moat",
            "risk_factors_only": "risk_audit",
        }.get(query_type)
        if required and artifacts.get(required) is None:
            logger.warning(f"Route {query_type} has no {required}; using the full report layout with notes.")
            return "full_10k_report"
        return query_type

    def _invoke_llm(
        self, context: str, instruction: str, callbacks: Optional[List[Any]]
    ) -> Tuple[Dict[str, Any], SynthesisProvenance]:
        messages = [SystemMessage(content=context), HumanMessage(content=instruction)]
        config = {"callbacks": callbacks} if callbacks else {}
        try:
            llm = self._get_llm()
            result = llm.with_structured_output(ResearchSynthesisPayload).invoke(messages, config=config)
            if isinstance(result, BaseModel):
                return result.model_dump(), "llm_structured"
            if isinstance(result, dict):
                return result, "llm_structured"
            raise ValueError(f"unexpected structured output type {type(result).__name__}")
        except Exception as e:
            logger.warning(f"Structured synthesis failed ({e!r}); retrying as plain JSON.")
        try:
            response = self._get_llm().invoke(messages, config=config)
            parsed = self._parse_json_fallback(response.content)
            if parsed:
                return parsed, "llm_json_fallback"
        except Exception as e:
            logger.error(f"Synthesis LLM fallback failed ({e!r}); using deterministic commentary.")
        return {}, "template_default"

    @staticmethod
    def _screen_prose(raw: Dict[str, Any], context: str, risk_audit: Optional[RiskAuditOutput]) -> Tuple[Dict[str, Optional[str]], int]:
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
            text = _drop_sentences(_strip_layout(text), _DATE_CLAIM)
            if no_risk_baseline and key in _RISK_PROSE:
                text = _drop_sentences(text, _YOY_CLAIM)
            text, removed = strip_unverified_sentences(text, known)
            removed_total += removed
            prose[key] = text or None
        if removed_total:
            logger.warning(f"Synthesis: removed {removed_total} sentence(s) citing figures absent from the context.")
        return prose, removed_total

    # ------------------------------------------------------------------
    # Deterministic fallback commentary (used per field when the LLM leaves it empty or fails)
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_prose(
        ticker: str,
        company_name: str,
        fiscal_year: int,
        business_moat: Optional[BusinessMoatOutput],
        financial_audit: Optional[FinancialAuditOutput],
        forecast: Optional[ForecastOutput],
        dcf_valuation: Optional[DCFValuationOutput],
        risk_audit: Optional[RiskAuditOutput],
    ) -> Dict[str, str]:
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cite_prose(
        prose: Dict[str, Optional[str]], fiscal_year: int, artifacts: Dict[str, Any], notes: List[str]
    ) -> Tuple[Any, Dict[str, Optional[str]]]:
        """Builds the report citer and appends source markers to verified prose; never raises."""
        try:
            citer = ReportCiter(CitationRegistry(), fiscal_year, **artifacts)
            index = build_figure_index(citer.figure_entries())
            cited = {
                key: annotate_sentences(text, index, citer.registry.cite_all) if text else text
                for key, text in prose.items()
            }
            return citer, cited
        except Exception as e:
            logger.exception(f"Citation build failed ({e!r}); publishing without filing markers.")
            notes.append("Filing citations could not be built for this report.")
            return NullCiter(), dict(prose)

    @staticmethod
    def _finalize_citations(citer: Any, markdown: str) -> Tuple[str, List[SourceRef]]:
        """Resolves and renumbers markers; on failure strips them, says so, and drops the evidence note."""
        if not isinstance(citer, ReportCiter):
            return strip_markers(markdown), []
        try:
            markdown, sources = citer.registry.finalize(markdown)
        except Exception as e:
            logger.exception(f"Citation finalize failed ({e!r}); publishing without filing markers.")
            markdown = strip_markers(markdown).replace(EVIDENCE_NOTE, NO_CITATIONS_NOTE)
            return markdown.rstrip("\n") + "\n\n" + CITATIONS_FAILED_NOTE + "\n", []
        if not any(s.kind != "market" for s in sources):
            markdown = markdown.replace(EVIDENCE_NOTE, NO_CITATIONS_NOTE)
        return markdown, sources

    @staticmethod
    def _parse_json_fallback(content: Any) -> Dict[str, Any]:
        """Extracts a JSON object from text content when structured output fails."""
        return StructuredAgent.parse_json_content(content)

    def run(
        self,
        messages: List[Dict[str, str]],
        session_state: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> AgentOutput:
        """The synthesizer is driven by the orchestrator with structured payloads; it has no chat mode."""
        last_user_msg = messages[-1].get("content", "") if messages else ""
        return AgentOutput(
            content=(
                "LeadSynthesizerAgent publishes reports from structured multi-agent payloads; "
                f"route research requests through the orchestrator ('{last_user_msg[:60]}')."
            ),
            sources=[],
        )


def write_research_report(run: RunContext, upstream: Upstream) -> Final10KResearchReport:
    """Pipeline entry: assembles the route's report from every research output that exists."""
    return LeadSynthesizerAgent().synthesize(
        ticker=run.ticker,
        company_name=run.company_name,
        fiscal_year=run.fiscal_year,
        business_moat=upstream.get("business_moat"),
        financial_audit=upstream.get("financial_audit"),
        forecast=upstream.get("forecast"),
        dcf_valuation=upstream.get("dcf_valuation"),
        risk_audit=upstream.get("risk_audit"),
        year_substituted=run.year_substituted,
        available_fiscal_years=list(run.available_fiscal_years),
        user_query=run.user_query,
        query_type=run.route,
        pipeline_notes=[note.message for note in upstream.notes],
        callbacks=run.callbacks,
    )


def summarize_report(report: Optional[Final10KResearchReport]) -> StatusLine:
    return "Synthesized 3-Pillar Thesis and compiled final publication report.", None

