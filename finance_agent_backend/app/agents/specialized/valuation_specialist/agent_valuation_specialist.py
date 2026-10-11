"""DCF Valuation Specialist Agent.

Deterministic-first: `build_valuation` derives the CAPM WACC, the 2-stage mid-year DCF (value-driver
terminal value, full equity bridge, roll-forward to the valuation date), the bull/base/bear scenario
values and the 5x5 sensitivity grid in Python. A single tool-free LLM call then writes the
institutional commentary; if it fails, a deterministic template is used instead.
"""

import logging
from datetime import date
from typing import Any, List, Optional

from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput
from app.agents.tools.valuation_model import MarketInputs, ValuationBuild, build_valuation


logger = logging.getLogger("finance_agent.agents.valuation_specialist")

COMMENTARY_MAX_TOKENS = 450
MIN_COMMENTARY_CHARS = 200


@AgentRegistry.register("valuation_specialist")
class ValuationSpecialistAgent(StructuredAgent[DCFValuationOutput]):
    """Code-built DCF valuation with one LLM call for commentary."""

    prompt_name = "valuation_specialist"
    tools = []
    output_schema = DCFValuationOutput

    def value(
        self,
        ticker: str,
        fiscal_year: int,
        financial_audit: FinancialAuditOutput,
        forecast: Optional[ForecastOutput],
        market: MarketInputs,
        terminal_growth_rate: float,
        wacc_override: Optional[float] = None,
        valuation_date: Optional[date] = None,
        balance_sheet_date: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> DCFValuationOutput:
        """Entry point for the LangGraph orchestrator and tests."""
        build = build_valuation(
            ticker,
            fiscal_year,
            financial_audit,
            forecast,
            market,
            terminal_growth_rate=terminal_growth_rate,
            wacc_override=wacc_override,
            valuation_date=valuation_date,
            balance_sheet_date=balance_sheet_date,
        )

        summary, source = None, "template"
        try:
            summary = self._run_commentary_llm(
                render_prompt("prompt_valuation_query.j2", v=build), callbacks
            ).strip()
            if len(summary) >= MIN_COMMENTARY_CHARS:
                source = "llm"
            else:
                logger.warning(f"Valuation commentary for {ticker} too short ({len(summary)} chars); using template.")
        except Exception as e:
            logger.warning(f"Valuation commentary LLM failed for {ticker}: {e}; using template.")

        if source == "template":
            summary = self._template_summary(build)
        return DCFValuationOutput(**build.model_dump(), valuation_summary=summary, commentary_source=source)

    def _run_commentary_llm(self, prompt: str, callbacks: Optional[List[Any]]) -> str:
        """Single tool-free completion that turns the code-built valuation into commentary."""
        from dotenv import load_dotenv
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        load_dotenv(override=True)
        llm = ChatOpenAI(
            model=self.model_name.replace("openai:", ""),
            temperature=0,
            max_retries=5,
            max_tokens=COMMENTARY_MAX_TOKENS,
        )
        config = {"callbacks": callbacks} if callbacks else {}
        messages = [SystemMessage(content=self.get_system_prompt()), HumanMessage(content=prompt)]
        return llm.invoke(messages, config=config).content

    @staticmethod
    def _template_summary(v: ValuationBuild) -> str:
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
