"""Lead Synthesizer & Institutional Report Publisher Agent.

Code-assembled report: the route-specific skeleton and every table (dashboard, audited statements and
Tables A–F, forecast schedule, WACC, equity bridge, scenarios, sensitivity, risk matrix and evidence) are
rendered in code from the specialist artifacts (one `report/<route>.j2` layout per route). One structured LLM call writes only
the analytical prose between them. Prose is screened in code: sentences citing figures absent from the
verified context are removed, invented dates and unsupported year-over-year risk claims are stripped,
and every field has a deterministic fallback, so a failed LLM call still yields a complete report.
Every filing-backed table row, quote and prose figure carries an inline [[n]](#cite-n) marker built in code
(report_citations.py); the numbered sources ship as Final10KResearchReport.sources.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agents.base import AgentOutput, BaseAgent, StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.citations import CitationRegistry, SourceRef, strip_markers
from app.agents.shared.figures import annotate_sentences, build_figure_index
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
from app.agents.specialized.lead_synthesizer.modules.report_coverage import assess_report_coverage
from app.agents.specialized.lead_synthesizer.write import (
    ResearchSynthesisPayload,
    SynthesisProvenance,
    fill_empty_prose,
    render_write_prompts,
    screen_prose,
    write_fallback_prose,
)

logger = logging.getLogger("finance_agent.agents.lead_synthesizer")

SYNTHESIS_MAX_TOKENS = 4000


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
        artifacts = dict(
            business_moat=business_moat,
            financial_audit=financial_audit,
            forecast=forecast,
            dcf_valuation=dcf_valuation,
            risk_audit=risk_audit,
        )
        coverage = assess_report_coverage(query_type, artifacts, pipeline_notes or [])
        query_type, notes = coverage.route, list(coverage.notes)

        context, instruction = render_write_prompts(
            ticker, company_name, fiscal_year, available_fiscal_years, query_type, user_query,
            year_substituted, notes, artifacts,
        )
        raw, provenance = self._invoke_llm(context, instruction, callbacks)
        prose, removed = screen_prose(raw, context, risk_audit)
        prose = fill_empty_prose(prose, write_fallback_prose(ticker, company_name, fiscal_year, **artifacts))

        citer, cited_prose = self._cite_prose(prose, fiscal_year, artifacts, notes)
        full_report_md = render_prompt(
            "lead_synthesizer", f"report/{coverage.layout}.j2",
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
    # LLM call (the write step's prompts, screen and fallback live in write.py)
    # ------------------------------------------------------------------
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

