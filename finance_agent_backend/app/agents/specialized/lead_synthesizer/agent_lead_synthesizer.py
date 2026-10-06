"""Lead Synthesizer & Institutional Report Publisher Agent.

Ingests structured payloads across all specialized domain agents (Business Strategist,
Financial Auditor, Forecaster, DCF Valuation Specialist, Risk Analyst), and orchestrates
the AI to dynamically generate the complete, publication-grade markdown response
tailored to the user's inquiry, with zero arithmetic hallucination and full provenance tracking.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.base import AgentOutput, BaseAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.registry import AgentRegistry
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.specialized.risk_analyst.state_risk_analyst import RiskAuditOutput
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput
from app.agents.specialized.lead_synthesizer.state_lead_synthesizer import (
    Final10KResearchReport,
    ThreePillarThesis,
)


logger = logging.getLogger("finance_agent.agents.lead_synthesizer")


class ResearchSynthesisPayload(BaseModel):
    """Internal Pydantic schema for structured LLM report and thesis generation."""

    full_markdown_report: str = Field(
        ...,
        description="The complete, publication-grade markdown response dynamically generated to directly answer the user inquiry, embedding verified tables.",
    )
    executive_summary: str = Field(
        ..., description="Executive briefing highlighting core findings and conclusion"
    )
    valuation_stance: Literal["Undervalued", "Fairly Valued", "Overvalued"] = Field(
        default="Fairly Valued", description="Valuation stance"
    )
    pillar_1_business_moat: Optional[str] = Field(
        None, description="Pillar 1: Business architecture and competitive moat durability (if applicable)"
    )
    pillar_2_financial_durability: Optional[str] = Field(
        None, description="Pillar 2: Earnings quality, balance sheet strength, and FCF conversion (if applicable)"
    )
    pillar_3_valuation_asymmetry: Optional[str] = Field(
        None, description="Pillar 3: Intrinsic value vs market price, margin of safety, and risk/reward asymmetry (if applicable)"
    )


@AgentRegistry.register("lead_synthesizer")
class LeadSynthesizerAgent(BaseAgent):
    """Institutional report publisher synthesizing multi-agent findings into a final publication."""

    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        self.model_name = model_name
        self._cached_llm = None

    def _get_llm(self):
        """Cache-first LLM resolver to eliminate redundant disk I/O from load_dotenv."""
        if self._cached_llm is not None:
            return self._cached_llm

        load_dotenv(override=True)
        model_clean = self.model_name.replace("openai:", "")
        llm = ChatOpenAI(
            model=model_clean,
            temperature=0,
            max_retries=5,
        )
        self._cached_llm = llm
        return llm

    @staticmethod
    def _fmt(
        val: Optional[float],
        fmt_spec: str = ".1f",
        scale: float = 1.0,
        prefix: str = "",
        suffix: str = "",
        default: str = "N/A",
    ) -> str:
        """Safely formats a floating-point metric, returning default if val is None, NaN, or infinite."""
        if val is None:
            return default
        try:
            import math
            f = float(val) * scale
            if math.isnan(f) or math.isinf(f):
                return default
            formatted = f"{f:{fmt_spec}}"
            return f"{prefix}{formatted}{suffix}"
        except (ValueError, TypeError):
            return default

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
        user_query: Optional[str] = None,
        query_type: str = "full_10k_report",
        callbacks: Optional[List[Any]] = None,
    ) -> Final10KResearchReport:
        """Synthesizes all structured agent payloads into a complete Final10KResearchReport."""
        llm = self._get_llm()

        # 1. Render prompt with context from all available domain agents
        prompt_text = render_prompt(
            "lead_synthesizer",
            ticker=ticker.upper(),
            company_name=company_name,
            fiscal_year=fiscal_year,
            business_moat=business_moat,
            financial_audit=financial_audit,
            forecast=forecast,
            dcf_valuation=dcf_valuation,
            risk_audit=risk_audit,
            query_type=query_type,
            user_query=user_query,
            year_substituted=year_substituted,
        )

        user_instruction = render_prompt(
            "prompt_synthesizer_instruction.j2",
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
            query_type=query_type,
            user_query=user_query,
        )

        # 2. Invoke LLM for qualitative synthesis and complete report generation
        raw_synthesis: Dict[str, Any] = {}
        synthesis_provenance: Literal["llm_structured", "llm_json_fallback", "template_default"] = "llm_structured"
        llm_config = {"callbacks": callbacks} if callbacks else {}

        try:
            structured_llm = llm.with_structured_output(ResearchSynthesisPayload)
            synthesis_obj = structured_llm.invoke([
                SystemMessage(content=prompt_text),
                HumanMessage(content=user_instruction),
            ], config=llm_config)
            if isinstance(synthesis_obj, BaseModel):
                raw_synthesis = synthesis_obj.model_dump()
                synthesis_provenance = "llm_structured"
            elif isinstance(synthesis_obj, dict):
                raw_synthesis = synthesis_obj
                synthesis_provenance = "llm_structured"
        except Exception as e:
            logger.warning(f"Structured output failed for lead synthesizer: {e}. Falling back to standard invoke.")
            try:
                resp = llm.invoke([
                    SystemMessage(content=prompt_text),
                    HumanMessage(content=user_instruction),
                ], config=llm_config)
                raw_synthesis = self._parse_json_fallback(resp.content)
                synthesis_provenance = "llm_json_fallback" if raw_synthesis else "template_default"
            except Exception as e2:
                logger.error(f"Fallback LLM invocation failed: {e2}. Defaulting to template strings.")
                raw_synthesis = {}
                synthesis_provenance = "template_default"

        # 3. Construct ThreePillarThesis with defensive defaults
        p1 = raw_synthesis.get("pillar_1_business_moat") or (
            "Competitive positioning and economic moat defensibility analyzed based on reported 10-K business operations."
            if business_moat else "Business model analyzed based on audited disclosures."
        )
        p2 = raw_synthesis.get("pillar_2_financial_durability") or (
            "Financial durability, operating cash flow generation, and balance sheet resilience evaluated from audited statements."
            if financial_audit else "Financial durability grounded in audited statements."
        )
        p3 = raw_synthesis.get("pillar_3_valuation_asymmetry") or (
            "Intrinsic DCF fair value offers risk-adjusted valuation perspective relative to prevailing market levels."
            if dcf_valuation else "Intrinsic valuation derived via discounted cash flow modeling."
        )

        thesis = ThreePillarThesis(
            pillar_1_business_moat=p1,
            pillar_2_financial_durability=p2,
            pillar_3_valuation_asymmetry=p3,
        )

        exec_summary = raw_synthesis.get("executive_summary") or (
            f"Institutional research evaluation of {company_name} ({ticker.upper()}) for Fiscal Year {fiscal_year}."
        )

        # 4. Resolve valuation stance & share price
        if dcf_valuation:
            fair_value = dcf_valuation.implied_fair_value_per_share
            current_price = dcf_valuation.current_share_price
            upside = dcf_valuation.upside_downside_pct
            stance = dcf_valuation.valuation_stance or raw_synthesis.get("valuation_stance", "Fairly Valued")
        else:
            fair_value = 0.0
            current_price = None
            upside = None
            stance = "Fairly Valued"

        if stance not in ("Undervalued", "Fairly Valued", "Overvalued"):
            stance = "Fairly Valued"

        # 5. Retrieve dynamic markdown report generated by the AI with fallback
        full_report_md = raw_synthesis.get("full_markdown_report")
        if not full_report_md:
            full_report_md = (
                f"# Research Analysis: {company_name} ({ticker.upper()})\n\n"
                f"> **Fiscal Year**: {fiscal_year} | **Valuation Stance**: **{stance.upper()}**\n\n"
                f"## Executive Summary\n{exec_summary}\n\n"
            )
            if dcf_valuation and dcf_valuation.sensitivity_matrix_markdown:
                full_report_md += f"## DCF Valuation & Sensitivity\n{dcf_valuation.sensitivity_matrix_markdown}\n"

        if year_substituted and "Filing Provenance Note" not in full_report_md:
            parts = full_report_md.split("\n", 1)
            prov_note = f"> ⚠️ **Filing Provenance Note**: Requested fiscal year unavailable in database catalog; substituted with latest audited filing (FY{fiscal_year}).\n"
            if len(parts) == 2:
                full_report_md = f"{parts[0]}\n{prov_note}\n{parts[1]}"
            else:
                full_report_md = f"{full_report_md}\n\n{prov_note}"

        if "Synthesis Provenance" not in full_report_md:
            full_report_md += f"\n\n> *Synthesis Provenance: `{synthesis_provenance}`*"

        # 6. Deduplicate citations
        all_citations = self._consolidate_citations(
            business_moat=business_moat,
            financial_audit=financial_audit,
            forecast=forecast,
            risk_audit=risk_audit,
        )

        return Final10KResearchReport(
            ticker=ticker.upper(),
            company_name=company_name,
            fiscal_year=fiscal_year,
            implied_fair_value_per_share=fair_value,
            current_share_price=current_price,
            upside_downside_pct=upside,
            valuation_stance=stance,
            three_pillar_thesis=thesis,
            executive_summary=exec_summary,
            synthesis_provenance=synthesis_provenance,
            year_substituted=year_substituted,
            full_markdown_report=full_report_md,
            all_citations=all_citations,
        )

    def _consolidate_citations(
        self,
        business_moat: Optional[BusinessMoatOutput],
        financial_audit: Optional[FinancialAuditOutput],
        forecast: Optional[ForecastOutput],
        risk_audit: Optional[RiskAuditOutput],
    ) -> List[Dict[str, Any]]:
        """Aggregates and deduplicates citation records across all sub-agents."""
        deduped = []
        seen = set()

        raw_lists = []
        if business_moat and business_moat.citations:
            raw_lists.append(business_moat.citations)
        if financial_audit and financial_audit.citations:
            raw_lists.append(financial_audit.citations)
        if forecast and forecast.citations:
            raw_lists.append(forecast.citations)
        if risk_audit and risk_audit.citations:
            raw_lists.append(risk_audit.citations)

        for lst in raw_lists:
            for item in lst:
                if isinstance(item, dict):
                    cid = item.get("chunk_id") or item.get("breadcrumb") or json.dumps(item)
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(item)

        return deduped

    def _parse_json_fallback(self, content: str) -> Dict[str, Any]:
        """Extracts JSON from text content when structured output fails."""
        clean = content.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean = "\n".join(lines).strip()
        try:
            return json.loads(clean)
        except Exception:
            return {}

    def run(
        self,
        messages: List[Dict[str, str]],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Executes the lead synthesizer on conversational messages conforming to BaseAgent."""
        last_user_msg = messages[-1].get("content", "") if messages else ""
        return AgentOutput(
            content=f"LeadSynthesizerAgent: Ready to publish institutional equity research report. "
                    f"Awaiting structured multi-agent state payloads for: '{last_user_msg[:60]}'.",
            sources=[],
        )
