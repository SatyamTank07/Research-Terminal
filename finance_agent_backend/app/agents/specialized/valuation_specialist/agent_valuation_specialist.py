"""DCF Valuation Specialist Agent: the checklist.

context (market quotes, balance-sheet date) -> six modules in MODULES order (cost_of_capital, intrinsic_value,
scenario_values, market_implied_expectations, sensitivity, valuation_stance) -> assemble -> write. Every number
is code-built; the write step makes the one tool-free LLM call for the commentary and falls back to a template.
"""

import logging
from datetime import date
from typing import Any, List, Optional

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.llm_call import complete_once
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.specialized.valuation_specialist.assemble import assemble_valuation
from app.agents.specialized.valuation_specialist.context import MarketInputs, balance_sheet_date, market_inputs
from app.agents.specialized.valuation_specialist.modules import (
    cost_of_capital,
    intrinsic_value,
    market_implied_expectations,
    scenario_values,
    sensitivity,
    valuation_stance,
)
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput, ValuationBuild
from app.agents.specialized.valuation_specialist.write import write_valuation_summary


logger = logging.getLogger("finance_agent.agents.valuation_specialist")

COMMENTARY_MAX_TOKENS = 650


def build_valuation(
    ticker: str,
    fiscal_year: int,
    financial_audit: FinancialAuditOutput,
    forecast: Optional[ForecastOutput],
    market: MarketInputs,
    *,
    terminal_growth_rate: float,
    wacc_override: Optional[float] = None,
    valuation_date: Optional[date] = None,
    balance_sheet_date: Optional[str] = None,
) -> ValuationBuild:
    """Every number of the valuation, module by module in MODULES order (no LLM)."""
    valuation_date = valuation_date or date.today()
    price = market.valid_share_price
    capital = cost_of_capital.estimate_cost_of_capital(
        cost_of_capital.gather_capital_inputs(financial_audit, market, wacc_override)
    )
    intrinsic = intrinsic_value.estimate_intrinsic_value(
        intrinsic_value.gather_intrinsic_value_inputs(financial_audit, forecast, market, valuation_date, balance_sheet_date),
        capital,
        terminal_growth_rate,
    )
    scenarios = scenario_values.value_scenarios(forecast, intrinsic, capital, price)
    implied = market_implied_expectations.solve_market_implied_expectations(forecast, intrinsic, capital, price)
    grids = sensitivity.build_sensitivity_grids(intrinsic.dcf)
    stance = valuation_stance.take_valuation_stance(intrinsic.dcf.fair_value_per_share, price)
    build = assemble_valuation(
        ticker=ticker, fiscal_year=fiscal_year, valuation_date=valuation_date,
        terminal_growth_rate=terminal_growth_rate, market=market, price=price, capital=capital,
        intrinsic=intrinsic, scenarios=scenarios, implied=implied, grids=grids, stance=stance,
    )
    logger.info(
        f"build_valuation {build.ticker} FY{fiscal_year}: WACC {capital.wacc_audit.wacc_pct:.2f}% "
        f"({capital.wacc_audit.wacc_source}), base ${build.implied_fair_value_per_share:.2f}, "
        f"bear ${build.bear_fair_value}, bull ${build.bull_fair_value}, "
        f"weighted ${build.probability_weighted_fair_value}, roll {intrinsic.terms.roll_forward_years:.2f}y, "
        f"{len(build.data_quality.warnings)} warnings"
    )
    return build


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

        summary, source = write_valuation_summary(build, lambda prompt: self._run_commentary_llm(prompt, callbacks))
        return DCFValuationOutput(**build.model_dump(), valuation_summary=summary, commentary_source=source)

    def _run_commentary_llm(self, prompt: str, callbacks: Optional[List[Any]]) -> str:
        """Single tool-free completion that turns the code-built valuation into commentary."""
        return complete_once(
            self.model_name, prompt, system_prompt=self.get_system_prompt(),
            max_tokens=COMMENTARY_MAX_TOKENS, max_retries=5, callbacks=callbacks,
        )


def value_equity(run: RunContext, upstream: Upstream) -> DCFValuationOutput:
    """Pipeline entry: CAPM WACC, DCF, scenarios and sensitivity at today's market inputs."""
    market = market_inputs(run.ticker, run.valuation_overrides)
    bs_date = balance_sheet_date(run.ticker, run.fiscal_year)
    user_wacc = run.valuation_overrides.get("wacc")
    wacc_label = f"{user_wacc * 100:.2f}%" if user_wacc is not None else "CAPM"
    logger.info(
        f"[valuation_specialist] Valuing {run.ticker} (price=${market.share_price} [{market.share_price_source}], "
        f"beta={market.beta} [{market.beta_source}], rf={market.risk_free_rate} [{market.risk_free_rate_source}], "
        f"g={run.terminal_growth_rate * 100:.1f}%, balance_sheet_date={bs_date}, user_wacc={wacc_label})"
    )
    return ValuationSpecialistAgent().value(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        financial_audit=upstream.get("financial_audit"),
        forecast=upstream.get("forecast"),
        market=market,
        terminal_growth_rate=run.terminal_growth_rate,
        wacc_override=user_wacc,
        valuation_date=run.valuation_date,
        balance_sheet_date=bs_date,
        callbacks=run.callbacks,
    )


def summarize_valuation(val: Optional[DCFValuationOutput]) -> StatusLine:
    if val is None:
        return "Valuation unavailable", None
    message = (
        f"Discounted at {val.wacc_audit.wacc_pct:.2f}% "
        f"({'user override' if val.wacc_audit.wacc_source == 'user_override' else 'CAPM WACC'}); "
        f"base-case DCF fair value ${val.implied_fair_value_per_share:.2f}/share"
    )
    if val.bear_fair_value is not None and val.bull_fair_value is not None:
        message += f" (bear ${val.bear_fair_value:.2f} - bull ${val.bull_fair_value:.2f})"
    return message, None
