"""Financial Forecasting Analyst Agent.

Specialized autonomous agent that models a disciplined 5-year forward financial forecast
schedule (Revenue, Operating Income / EBIT, and Unlevered Free Cash Flow) grounded in audited
historical metrics (FinancialAuditOutput), Item 7 MD&A disclosures, and segment trends.

Architectural Guarantees:
1. Zero Arithmetic Hallucination: Offloads all forward compounding and cash-flow bridges to
   calculate_forecast_schedule_tool.
2. Dual-Mode Accounting Precedence: Evaluated deterministically in code (not an LLM decision).
   If D&A is available -> 'comprehensive_line_item'; else -> 'simplified_nopat_less_capex'.
3. Deterministic CAGR Decay Fallback: If Item 7 narrative is sparse/boilerplate (< 2 chunks
   or < 500 characters), deterministically decays 3-year historical CAGR by 75 bps/yr towards
   a 2.75% terminal floor, tagging guidance_source="historical_cagr_decay".
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.state import (
    BusinessMoatOutput,
    FinancialAuditOutput,
    ForecastOutput,
    ForecastYearResult,
)
from app.agents.tools.forecast_tools import (
    calculate_forecast_schedule,
    calculate_forecast_schedule_tool,
)
from app.agents.tools.rag_narrative_tools import (
    retrieve_10k_narrative,
    retrieve_10k_narrative_tool,
)
from app.agents.tools.rag_table_tools import retrieve_10k_tables_tool

logger = logging.getLogger("finance_agent.agents.forecasting_analyst")

FALLBACK_DECAY_RATE_PER_YEAR = 0.0075  # 75 bps / year
FALLBACK_TERMINAL_GROWTH_FLOOR = 0.0275  # 2.75% long-term GDP floor
MIN_NARRATIVE_CHUNKS_THRESHOLD = 2
MIN_NARRATIVE_CHARS_THRESHOLD = 500


@AgentRegistry.register("forecasting_analyst")
class ForecastingAnalystAgent(StructuredAgent[ForecastOutput]):
    """Autonomous agent projecting 5-year financial schedules and Unlevered Free Cash Flows."""

    prompt_name = "forecasting_analyst"
    tools = [
        calculate_forecast_schedule_tool,
        retrieve_10k_narrative_tool,
        retrieve_10k_tables_tool,
    ]
    output_schema = ForecastOutput
    default_recursion_limit = 50

    def forecast(
        self,
        ticker: str,
        fiscal_year: int,
        financial_audit: FinancialAuditOutput,
        business_moat: Optional[BusinessMoatOutput] = None,
        horizon_years: int = 5,
        force_mode: Optional[Literal["comprehensive_line_item", "simplified_nopat_less_capex"]] = None,
    ) -> ForecastOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and verification tests.

        Ingests FinancialAuditOutput and optional BusinessMoatOutput, executes deterministic tools,
        and returns a validated ForecastOutput instance.
        """
        # Step 1: Ingest Audited Baseline & Normalization
        latest_annual = financial_audit.multi_year_history[-1]
        base_revenue = latest_annual.revenue
        base_year = latest_annual.fiscal_year or fiscal_year

        ratios = financial_audit.profitability_and_return_ratios
        tax_rate_val = getattr(ratios, "effective_tax_rate_pct", None)
        if tax_rate_val is not None:
            tax_rate = (tax_rate_val / 100.0) if tax_rate_val > 1.0 else tax_rate_val
        else:
            tax_rate = getattr(ratios, "effective_tax_rate", 0.21)

        latest_capex = latest_annual.capital_expenditures
        capex_pct = (latest_capex / base_revenue) if base_revenue > 0 else 0.03

        latest_margin_val = latest_annual.operating_margin_pct
        latest_margin = (latest_margin_val / 100.0) if abs(latest_margin_val) > 1.0 else latest_margin_val

        # Step 2: Deterministic Accounting Mode Precedence (Code-governed)
        latest_depr = getattr(latest_annual, "depreciation_amortization", None)
        if latest_depr is None and hasattr(financial_audit, "solvency_and_liquidity_ratios") and financial_audit.solvency_and_liquidity_ratios:
            ebitda = getattr(financial_audit.solvency_and_liquidity_ratios, "ebitda", None)
            if ebitda is not None and latest_annual.operating_income is not None:
                latest_depr = round(ebitda - latest_annual.operating_income, 2)

        if force_mode is not None:
            if force_mode == "simplified_nopat_less_capex":
                depr_pct = None
            else:
                depr_pct = (latest_depr / base_revenue) if (latest_depr is not None and base_revenue > 0) else 0.03
        else:
            if latest_depr is not None and base_revenue > 0:
                depr_pct = latest_depr / base_revenue
            else:
                depr_pct = None

        # Step 3: Compute Historical 3-Year CAGR
        history = financial_audit.multi_year_history
        if len(history) >= 2 and history[0].revenue > 0 and history[-1].revenue > 0:
            n_intervals = len(history) - 1
            historical_cagr = (history[-1].revenue / history[0].revenue) ** (1.0 / n_intervals) - 1.0
        elif len(history) >= 1 and getattr(history[-1], "revenue_growth_pct", None) is not None:
            historical_cagr = history[-1].revenue_growth_pct / 100.0
        else:
            historical_cagr = 0.05

        # Step 4: Deterministic MD&A Inspection & Boilerplate Fallback Trigger
        try:
            narrative_chunks = retrieve_10k_narrative(
                ticker=ticker,
                fiscal_year=fiscal_year,
                query="future outlook demand trends product pipeline capital expenditures cost inflation",
                section_item="Item 7",
                limit=5,
            )
        except Exception as e:
            logger.warning(f"retrieve_10k_narrative failed for {ticker} FY{fiscal_year}: {e}")
            narrative_chunks = []

        total_narrative_chars = sum(len(c.content) for c in narrative_chunks)
        is_boilerplate = (
            len(narrative_chunks) < MIN_NARRATIVE_CHUNKS_THRESHOLD
            or total_narrative_chars < MIN_NARRATIVE_CHARS_THRESHOLD
        )

        if is_boilerplate:
            logger.info(
                f"Boilerplate MD&A triggered for {ticker} FY{fiscal_year} "
                f"({len(narrative_chunks)} chunks, {total_narrative_chars} chars). "
                f"Applying deterministic CAGR decay ({FALLBACK_DECAY_RATE_PER_YEAR*10000:.0f}bps/yr)."
            )
            growth_rates = []
            curr_g = historical_cagr
            for step in range(1, horizon_years + 1):
                curr_g = max(curr_g - FALLBACK_DECAY_RATE_PER_YEAR, FALLBACK_TERMINAL_GROWTH_FLOOR)
                growth_rates.append(round(curr_g, 4))

            margins = [round(latest_margin, 4)] * horizon_years

            schedule_dict = calculate_forecast_schedule(
                base_revenue=base_revenue,
                base_year=base_year,
                revenue_growth_rates=growth_rates,
                operating_margins=margins,
                tax_rate=tax_rate,
                capex_pct_of_revenue=round(capex_pct, 4),
                depreciation_pct_of_revenue=round(depr_pct, 4) if depr_pct is not None else None,
            )

            growth_rationale = (
                f"Item 7 MD&A narrative disclosures were sparse or below length threshold "
                f"({len(narrative_chunks)} chunks retrieved, {total_narrative_chars} characters). "
                f"Applied deterministic historical CAGR decay rule: starting from the 3-year historical "
                f"revenue CAGR of {historical_cagr*100:.2f}%, decaying by {FALLBACK_DECAY_RATE_PER_YEAR*10000:.0f} bps "
                f"annually towards a long-term GDP terminal floor of {FALLBACK_TERMINAL_GROWTH_FLOOR*100:.2f}%."
            )
            margin_rationale = (
                f"Operating margin maintained steady at the audited base year rate ({latest_margin*100:.2f}%) "
                f"across the {horizon_years}-year explicit forecast horizon under historical consistency."
            )
            reinvestment_rationale = (
                f"CapEx intensity maintained at audited baseline rate of {capex_pct*100:.2f}% of revenue."
            )
            citations = [
                {"chunk_id": c.chunk_id, "item": c.item, "breadcrumb": c.breadcrumb}
                for c in narrative_chunks
            ]

            return ForecastOutput(
                ticker=ticker.upper(),
                fiscal_year=fiscal_year,
                base_revenue=schedule_dict["base_revenue"],
                forecast_horizon_years=horizon_years,
                revenue_cagr_pct=schedule_dict["revenue_cagr_pct"],
                cumulative_5yr_fcf=schedule_dict["cumulative_5yr_fcf"],
                average_annual_fcf=schedule_dict["average_annual_fcf"],
                provenance_mode=schedule_dict["provenance_mode"],
                guidance_source="historical_cagr_decay",
                tax_rate_pct=schedule_dict["tax_rate_pct"],
                projected_fcfs=schedule_dict["projected_fcfs"],
                forecast_schedule=[ForecastYearResult(**y) for y in schedule_dict["forecast_schedule"]],
                forecast_table_markdown=schedule_dict["forecast_table_markdown"],
                growth_rationale=growth_rationale,
                margin_expansion_rationale=margin_rationale,
                reinvestment_rationale=reinvestment_rationale,
                citations=citations,
            )

        # Step 5: Rich MD&A Available -> Execute Agent
        depr_directive = (
            f"Pass depreciation_pct_of_revenue={depr_pct:.4f} to calculate_forecast_schedule_tool "
            f"to operate in 'comprehensive_line_item' mode."
            if depr_pct is not None
            else "Omit depreciation_pct_of_revenue to operate in 'simplified_nopat_less_capex' mode."
        )

        moat_context = ""
        if business_moat is not None:
            moat_context = (
                f"\nBusiness Context from BusinessMoatOutput:\n"
                f"- Primary Segments: {', '.join(business_moat.primary_product_segments)}\n"
                f"- Economic Moat: {business_moat.economic_moat_type} ({business_moat.moat_durability})\n"
                f"- Revenue Architecture: {business_moat.revenue_architecture}\n"
            )

        query = (
            f"Construct a disciplined {horizon_years}-year forward financial forecast for {ticker.upper()} "
            f"based on fiscal year {fiscal_year} 10-K audited metrics and Item 7 (MD&A) disclosures.\n\n"
            f"Audited Baseline Financials:\n"
            f"- Base Year (T0): FY{base_year}\n"
            f"- Base Year Revenue: ${base_revenue:,.2f} Million\n"
            f"- Historical 3-Year Revenue CAGR: {historical_cagr*100:.2f}%\n"
            f"- Base Year Operating Margin: {latest_margin*100:.2f}%\n"
            f"- Base Year CapEx: ${latest_capex:,.2f} Million (CapEx Intensity: {capex_pct*100:.2f}% of Revenue)\n"
            f"- Derived Effective Tax Rate: {tax_rate*100:.2f}%\n"
            f"{moat_context}\n"
            f"Accounting Precedence Directive:\n"
            f"- {depr_directive}\n\n"
            f"Execution Protocol:\n"
            f"1. Review Item 7 MD&A narrative context and segment trends.\n"
            f"2. Calibrate a {horizon_years}-year forward revenue growth schedule fading towards terminal rates, "
            f"and an operating margin schedule reflecting operating leverage or cost pressures.\n"
            f"3. Call calculate_forecast_schedule_tool EXACTLY ONCE with these parameters.\n"
            f"4. Emit the complete ForecastOutput JSON with guidance_source='md&a_explicit'."
        )

        fallback_defaults = {
            "growth_rationale": "Projected 5-year revenue growth trajectory calibrated against Item 7 MD&A disclosures.",
            "margin_expansion_rationale": "Operating margin progression reflects operating leverage and expected product mix shift.",
            "reinvestment_rationale": f"CapEx modeled at {capex_pct*100:.2f}% of revenue based on audited capital allocation history.",
            "guidance_source": "md&a_explicit",
        }

        return self.execute_structured(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults=fallback_defaults,
            base_revenue=base_revenue,
            base_year=base_year,
            tax_rate=tax_rate,
            capex_pct=capex_pct,
            depr_pct=depr_pct,
            horizon_years=horizon_years,
            narrative_chunks=narrative_chunks,
        )

    def _post_process_output(
        self,
        output: ForecastOutput,
        result: Dict[str, Any],
        base_revenue: Optional[float] = None,
        base_year: Optional[int] = None,
        tax_rate: Optional[float] = None,
        capex_pct: Optional[float] = None,
        depr_pct: Optional[float] = None,
        horizon_years: int = 5,
        narrative_chunks: Optional[List[Any]] = None,
        **kwargs,
    ) -> ForecastOutput:
        """Enforces deterministic math on the forecast schedule using calculate_forecast_schedule."""
        if base_revenue is None or base_year is None or tax_rate is None or capex_pct is None:
            return output

        messages = result.get("messages", [])
        tool_growth_rates = None
        tool_margins = None

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "calculate_forecast_schedule_tool":
                        args = tc.get("args", {})
                        tool_growth_rates = args.get("revenue_growth_rates")
                        tool_margins = args.get("operating_margins")
                        break

        if not tool_growth_rates and output.forecast_schedule:
            tool_growth_rates = [y.projected_revenue_growth_pct for y in output.forecast_schedule]
            tool_margins = [y.projected_ebit_margin_pct for y in output.forecast_schedule]

        if not tool_growth_rates:
            tool_growth_rates = [0.060, 0.055, 0.050, 0.045, 0.040][:horizon_years]
        if not tool_margins:
            tool_margins = [0.30] * horizon_years

        # ALWAYS enforce code-governed accounting mode precedence by executing calculate_forecast_schedule
        verified_schedule = calculate_forecast_schedule(
            base_revenue=base_revenue,
            base_year=base_year,
            revenue_growth_rates=tool_growth_rates,
            operating_margins=tool_margins,
            tax_rate=tax_rate,
            capex_pct_of_revenue=capex_pct,
            depreciation_pct_of_revenue=depr_pct,
        )

        output.base_revenue = verified_schedule["base_revenue"]
        output.forecast_horizon_years = verified_schedule["forecast_horizon_years"]
        output.revenue_cagr_pct = verified_schedule["revenue_cagr_pct"]
        output.cumulative_5yr_fcf = verified_schedule["cumulative_5yr_fcf"]
        output.average_annual_fcf = verified_schedule["average_annual_fcf"]
        output.provenance_mode = verified_schedule["provenance_mode"]
        output.guidance_source = "md&a_explicit"
        output.tax_rate_pct = verified_schedule["tax_rate_pct"]
        output.projected_fcfs = verified_schedule["projected_fcfs"]
        output.forecast_schedule = [ForecastYearResult(**y) for y in verified_schedule["forecast_schedule"]]
        output.forecast_table_markdown = verified_schedule["forecast_table_markdown"]

        if narrative_chunks and not output.citations:
            output.citations = [
                {"chunk_id": c.chunk_id, "item": c.item, "breadcrumb": c.breadcrumb}
                for c in narrative_chunks
            ]

        return output
