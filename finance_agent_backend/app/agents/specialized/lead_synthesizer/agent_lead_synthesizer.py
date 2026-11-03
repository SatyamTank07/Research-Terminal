"""Lead Synthesizer: the checklist that turns every research output of the route into its report.

report_coverage (which layout, which run notes) -> write (write.py: one structured LLM call writes only the prose
fields; code screens them and backs each one up) -> assemble (assemble.py: source markers on the verified prose,
the route's report/<layout>.j2 with every table rendered in code from the artifacts, the numbered sources).
Every filing-backed table row, quote and prose figure carries an inline [[n]](#cite-n) marker (report_citations.py);
the numbered sources ship as Final10KResearchReport.sources.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agents.base import AgentOutput, BaseAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.llm_call import parse_json_content
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.specialized.lead_synthesizer.assemble import (
    assemble_report,
    cite_prose,
    finalize_citations,
    render_report,
)
from app.agents.specialized.lead_synthesizer.modules.report_coverage import assess_report_coverage
from app.agents.specialized.lead_synthesizer.state_lead_synthesizer import Final10KResearchReport
from app.agents.specialized.lead_synthesizer.write import (
    ResearchSynthesisPayload,
    SynthesisProvenance,
    fill_empty_prose,
    render_write_prompts,
    screen_prose,
    write_fallback_prose,
)
from app.agents.specialized.risk_analyst import RiskAuditOutput
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput

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
        """The checklist: report coverage -> write the prose -> cite, render and assemble the report."""
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
            ticker=ticker, company_name=company_name, fiscal_year=fiscal_year,
            available_fiscal_years=available_fiscal_years, query_type=query_type, user_query=user_query,
            year_substituted=year_substituted, pipeline_notes=notes, artifacts=artifacts,
        )
        raw, provenance = self._invoke_llm(context, instruction, callbacks)
        prose, removed = screen_prose(raw, context, risk_audit)
        prose = fill_empty_prose(prose, write_fallback_prose(ticker, company_name, fiscal_year, **artifacts))

        citer, cited_prose = cite_prose(prose, fiscal_year, artifacts, notes)
        markdown = render_report(
            layout=coverage.layout, ticker=ticker, company_name=company_name, fiscal_year=fiscal_year,
            year_substituted=year_substituted, prose=cited_prose, notes=notes, provenance=provenance,
            removed_sentences=removed, citer=citer, artifacts=artifacts,
        )
        markdown, sources = finalize_citations(citer, markdown)
        return assemble_report(
            ticker=ticker, company_name=company_name, fiscal_year=fiscal_year, dcf_valuation=dcf_valuation,
            prose=prose, provenance=provenance, year_substituted=year_substituted, markdown=markdown, sources=sources,
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
            parsed = parse_json_content(response.content)
            if parsed:
                return parsed, "llm_json_fallback"
        except Exception as e:
            logger.error(f"Synthesis LLM fallback failed ({e!r}); using deterministic commentary.")
        return {}, "template_default"

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

