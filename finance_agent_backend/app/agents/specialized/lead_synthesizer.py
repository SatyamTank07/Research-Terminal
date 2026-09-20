"""Lead Synthesizer & Institutional Report Publisher Agent.

Ingests structured payloads across all specialized domain agents (Business Strategist,
Financial Auditor, Forecaster, DCF Valuation Specialist, Risk Analyst), formulates the
3-Pillar Investment Thesis and Executive Summary via LLM, and deterministically compiles
the complete institutional-grade Markdown research publication with zero math hallucination.
Exhaustively guards all numerical values against None/NaN/Inf and tags synthesis provenance.
"""

import json
import logging
import math
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.base import AgentOutput, BaseAgent
from app.agents.prompts import render_prompt
from app.agents.registry import AgentRegistry
from app.agents.state import (
    BusinessMoatOutput,
    DCFValuationOutput,
    Final10KResearchReport,
    FinancialAuditOutput,
    ForecastOutput,
    RiskAuditOutput,
    ThreePillarThesis,
)

logger = logging.getLogger("finance_agent.agents.lead_synthesizer")


class ResearchSynthesisPayload(BaseModel):
    """Internal Pydantic schema for structured LLM thesis and summary generation."""

    pillar_1_business_moat: str = Field(
        ..., description="Pillar 1: Business architecture, competitive moat durability, and pricing power"
    )
    pillar_2_financial_durability: str = Field(
        ..., description="Pillar 2: Earnings quality, balance sheet strength, FCF conversion, and ROIC vs WACC"
    )
    pillar_3_valuation_asymmetry: str = Field(
        ..., description="Pillar 3: Intrinsic value vs market price, margin of safety, and risk/reward asymmetry"
    )
    executive_summary: str = Field(
        ..., description="Executive briefing highlighting business model, growth, and risks"
    )
    valuation_stance: Literal["Undervalued", "Fairly Valued", "Overvalued"] = Field(
        default="Fairly Valued", description="Valuation stance"
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
        )

        user_instruction = (
            f"Synthesize the research findings for {ticker.upper()} (FY{fiscal_year}). "
            f"Formulate the 3-Pillar Investment Thesis (Pillar 1: Business Moat, "
            f"Pillar 2: Financial Durability, Pillar 3: Valuation Asymmetry), "
            f"the executive summary, and the final valuation stance."
        )

        # 2. Invoke LLM for qualitative synthesis with provenance tracking
        raw_synthesis: Dict[str, Any] = {}
        synthesis_provenance: Literal["llm_structured", "llm_json_fallback", "template_default"] = "llm_structured"

        try:
            structured_llm = llm.with_structured_output(ResearchSynthesisPayload)
            synthesis_obj = structured_llm.invoke([
                SystemMessage(content=prompt_text),
                HumanMessage(content=user_instruction),
            ])
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
                ])
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

        # 5. Compile full institutional Markdown publication with None-safe formatting
        full_report_md = self._build_markdown_report(
            ticker=ticker.upper(),
            company_name=company_name,
            fiscal_year=fiscal_year,
            thesis=thesis,
            exec_summary=exec_summary,
            fair_value=fair_value,
            current_price=current_price,
            upside=upside,
            stance=stance,
            business_moat=business_moat,
            financial_audit=financial_audit,
            forecast=forecast,
            dcf_valuation=dcf_valuation,
            risk_audit=risk_audit,
            year_substituted=year_substituted,
            synthesis_provenance=synthesis_provenance,
        )

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

    def _build_markdown_report(
        self,
        ticker: str,
        company_name: str,
        fiscal_year: int,
        thesis: ThreePillarThesis,
        exec_summary: str,
        fair_value: float,
        current_price: Optional[float],
        upside: Optional[float],
        stance: str,
        business_moat: Optional[BusinessMoatOutput],
        financial_audit: Optional[FinancialAuditOutput],
        forecast: Optional[ForecastOutput],
        dcf_valuation: Optional[DCFValuationOutput],
        risk_audit: Optional[RiskAuditOutput],
        year_substituted: bool = False,
        synthesis_provenance: str = "llm_structured",
    ) -> str:
        """Deterministically stitches the complete publication report with exhaustive None-guarding."""
        lines = []

        # Title Block & Filing Source
        lines.append(f"# Institutional Equity Research Report: {company_name} ({ticker})")
        lines.append(f"> **Filing Source**: SEC Form 10-K (Fiscal Year {fiscal_year}) | **Valuation Stance**: **{stance.upper()}**")
        if year_substituted:
            lines.append(f"> ⚠️ **Filing Provenance Note**: Requested fiscal year unavailable in database catalog; substituted with latest audited filing (FY{fiscal_year}).")
        lines.append("")

        # Executive Valuation Dashboard Card
        lines.append("## Executive Valuation Dashboard")
        lines.append("| Metric | Value | Provenance / Convention |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| **Implied DCF Fair Value** | **{self._fmt(fair_value, '.2f', prefix='$')}** | 2-Stage Gordon Growth (Mid-Year Discounting) |")
        if current_price:
            lines.append(f"| **Current Market Price** | {self._fmt(current_price, '.2f', prefix='$')} | Latest Quoted Close |")
            lines.append(f"| **Implied Margin of Safety / Upside** | **{self._fmt(upside, '+.1f', suffix='%')}** | Relative to Implied Fair Value |")
        lines.append(f"| **Valuation Verdict** | **{stance}** | Threshold: ±10% Margin of Safety |")
        if dcf_valuation:
            lines.append(f"| **Blended WACC Hurdle** | {self._fmt(dcf_valuation.wacc_audit.wacc_pct, '.2f', suffix='%')} | CAPM Ke: {self._fmt(dcf_valuation.wacc_audit.cost_of_equity_pct, '.2f', suffix='%')} |")
            lines.append(f"| **Terminal Growth Rate (g)** | {self._fmt(dcf_valuation.terminal_growth_rate, '.1f', scale=100.0, suffix='%')} | Long-Term GDP Baseline |")
            lines.append(f"| **Terminal Value % of EV** | {self._fmt(dcf_valuation.terminal_value_pct_of_ev, '.1f', suffix='%')} | Standard Bound: 60%–80% |")
            lines.append(f"| **Enterprise Value (EV)** | {self._fmt(dcf_valuation.enterprise_value, ',.1f', prefix='$', suffix='M')} | PV Explicit UFCFs + PV Terminal Value |")
            
            if dcf_valuation.net_debt is None:
                net_debt_desc = "N/A"
            elif dcf_valuation.net_debt <= 0:
                net_debt_desc = "Net Cash Surplus"
            else:
                net_debt_desc = "Net Debt Drag"
            lines.append(f"| **Net Debt** | {self._fmt(dcf_valuation.net_debt, ',.1f', prefix='$', suffix='M')} | {net_debt_desc} |")
            lines.append(f"| **Diluted Shares Outstanding** | {self._fmt(dcf_valuation.diluted_shares, ',.1f', suffix='M')} | Audited 10-K Item 8 Balance Sheet |")
        lines.append("")

        # Section 1: Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append(exec_summary)
        lines.append("")

        # Section 2: The 3-Pillar Investment Thesis
        lines.append("## 2. Institutional 3-Pillar Investment Thesis")
        lines.append(f"### Pillar 1: Business Architecture & Competitive Moat\n{thesis.pillar_1_business_moat}\n")
        lines.append(f"### Pillar 2: Financial Durability & Earnings Quality\n{thesis.pillar_2_financial_durability}\n")
        lines.append(f"### Pillar 3: Valuation Asymmetry & Margin of Safety\n{thesis.pillar_3_valuation_asymmetry}\n")

        # Section 3: Business Operations & Moat Analysis
        if business_moat:
            lines.append("## 3. Business Operations & Economic Moat Analysis")
            lines.append(f"**Business Overview**: {business_moat.business_summary}\n")
            lines.append(f"**Revenue Architecture**: {business_moat.revenue_architecture}\n")
            lines.append(f"- **Economic Moat**: **{business_moat.economic_moat_type}** ({business_moat.moat_durability} Durability, {business_moat.moat_trajectory} Trajectory)")
            lines.append(f"- **Moat Defense**: {business_moat.moat_rationale}")
            lines.append(f"- **Pricing Power**: {business_moat.pricing_power_assessment}")
            lines.append(f"- **Customer Concentration**: {business_moat.customer_concentration}\n")

            if business_moat.segment_details:
                lines.append("### Primary Operating Segments")
                for seg in business_moat.segment_details:
                    drivers_str = f" (*Key Drivers*: {', '.join(seg.growth_drivers)})" if seg.growth_drivers else ""
                    lines.append(f"- **{seg.name}**: {seg.description}{drivers_str}")
                lines.append("")

        # Section 4: Audited Financial Statements & Ratio Performance
        if financial_audit:
            lines.append("## 4. Audited Financial Statements & Ratio Performance")
            lines.append(f"*{financial_audit.auditor_summary}*\n")

            # Ratio performance table with exhaustive None-guarding
            lines.append("### Key Financial Ratios & Return Metrics")
            lines.append("| Metric | Audited Value | Benchmark / Interpretation |")
            lines.append("| :--- | :--- | :--- |")
            r = financial_audit.profitability_and_return_ratios
            s = financial_audit.solvency_and_liquidity_ratios
            lines.append(f"| **ROIC** | **{self._fmt(r.roic_pct, '.1f', suffix='%')}** | After-Tax Return on Invested Capital |")
            lines.append(f"| **Effective Tax Rate** | {self._fmt(r.effective_tax_rate_pct, '.1f', suffix='%')} | Audited 10-K Effective Rate |")
            net_debt_str = self._fmt(s.net_debt_to_ebitda, '.2f', suffix='x')
            benchmark_str = 'Conservative / Cash Surplus' if (s.net_debt_to_ebitda is not None and s.net_debt_to_ebitda <= 0) else 'Leveraged'
            lines.append(f"| **Net Debt / EBITDA** | {net_debt_str} | {benchmark_str} |")
            lines.append(f"| **Current Ratio** | {self._fmt(s.current_ratio, '.2f', suffix='x')} | Liquidity Coverage |")
            lines.append("")

            # Forensic red flags
            if financial_audit.forensic_red_flags:
                lines.append("### Forensic Accounting Audit Checks")
                for flag in financial_audit.forensic_red_flags:
                    lines.append(f"- ⚠️ **Flag**: {flag}")
                lines.append("")

            # Raw markdown statement tables
            if financial_audit.income_statement_markdown_table:
                lines.append("### Audited Consolidated Statements of Operations")
                lines.append(financial_audit.income_statement_markdown_table)
                lines.append("")

            if financial_audit.balance_sheet_markdown_table:
                lines.append("### Audited Consolidated Balance Sheets")
                lines.append(financial_audit.balance_sheet_markdown_table)
                lines.append("")

            if financial_audit.cash_flow_markdown_table:
                lines.append("### Audited Consolidated Statements of Cash Flows")
                lines.append(financial_audit.cash_flow_markdown_table)
                lines.append("")

        # Section 5: 5-Year Forecast Schedule
        if forecast:
            lines.append("## 5. 5-Year Financial Forecast & Cash Flow Schedule")
            lines.append(f"- **5-Year Revenue CAGR**: **{self._fmt(forecast.revenue_cagr_pct, '.1f', suffix='%')}**")
            lines.append(
                f"- **5-Year Cumulative UFCF**: **{self._fmt(forecast.cumulative_5yr_fcf, ',.1f', prefix='$', suffix='M')}** "
                f"(Annual Avg: {self._fmt(forecast.average_annual_fcf, ',.1f', prefix='$', suffix='M')})"
            )
            lines.append(f"- **Guidance Provenance**: `{forecast.guidance_source}` ({forecast.provenance_mode})")
            lines.append(f"- **Growth Rationale**: {forecast.growth_rationale}")
            lines.append(f"- **Margin Expansion Rationale**: {forecast.margin_expansion_rationale}\n")

            if forecast.forecast_table_markdown:
                lines.append(forecast.forecast_table_markdown)
                lines.append("")

        # Section 6: DCF Valuation & Sensitivity Analysis
        if dcf_valuation:
            lines.append("## 6. Discounted Cash Flow (DCF) Valuation & Sensitivity")
            lines.append(f"*{dcf_valuation.valuation_summary}*\n")

            lines.append("### WACC Hurdle Rate Breakdown")
            lines.append(dcf_valuation.wacc_audit.formula_breakdown_markdown)
            lines.append("")

            lines.append("### 2-Way Sensitivity Matrix: WACC vs. Perpetual Terminal Growth")
            lines.append(dcf_valuation.sensitivity_matrix_markdown)
            lines.append("")

        # Section 7: Risk Factors & Existential Overhangs
        if risk_audit:
            lines.append("## 7. Material Risk Factors & Existential Overhangs")
            lines.append(f"- **Overall Risk Rating**: **{risk_audit.overall_risk_profile}**")
            lines.append(f"- **Primary Existential Threat**: {risk_audit.primary_existential_threat}\n")

            lines.append("| Severity | Category | Risk Title & 10-K Item 1A Disclosure |")
            lines.append("| :--- | :--- | :--- |")
            for rk in risk_audit.identified_risks:
                lines.append(f"| **{rk.severity}** | {rk.risk_category} | **{rk.risk_title}**: {rk.risk_summary} |")
            lines.append("")

        # Section 8: Compliance & Provenance Disclaimers
        lines.append("## 8. Regulatory Disclaimers & Audit Breadcrumbs")
        lines.append(
            f"> *Disclaimer: This equity research report is generated automatically by the Antigravity Multi-Agent Research System "
            f"based strictly on audited SEC Form 10-K filings. Synthesis Provenance: `{synthesis_provenance}`. "
            f"All financial ratios, discount rates, and discounted cash flows are derived deterministically "
            f"via verified Python calculation engines. Not financial advice.*"
        )
        lines.append("")

        return "\n".join(lines)

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

    def run(self, messages: List[Dict[str, str]]) -> AgentOutput:
        """Executes the lead synthesizer on conversational messages conforming to BaseAgent."""
        last_user_msg = messages[-1].get("content", "") if messages else ""
        return AgentOutput(
            content=f"LeadSynthesizerAgent: Ready to publish institutional equity research report. "
                    f"Awaiting structured multi-agent state payloads for: '{last_user_msg[:60]}'.",
            sources=[],
        )
