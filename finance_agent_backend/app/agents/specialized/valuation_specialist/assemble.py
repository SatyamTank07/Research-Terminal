"""Builds every number of the valuation (no LLM). Tasks of PR 4 carve the analyst-question modules out of
this file; what remains assembles their results into ValuationBuild.

Consumes the Financial Auditor and Forecasting Analyst artifacts plus resolved market inputs. All `_pct` fields
from upstream agents are whole percents and are converted to decimals here exactly once.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, List, Optional

from app.agents.specialized.valuation_specialist.context import MarketInputs
from app.agents.specialized.valuation_specialist.modules.cost_of_capital import (
    estimate_cost_of_capital,
    gather_capital_inputs,
)
from app.agents.specialized.valuation_specialist.modules.intrinsic_value import (
    estimate_intrinsic_value,
    gather_intrinsic_value_inputs,
)
from app.agents.specialized.valuation_specialist.modules.market_implied_expectations import solve_market_implied_expectations
from app.agents.specialized.valuation_specialist.modules.scenario_values import value_scenarios
from app.agents.specialized.valuation_specialist.modules.sensitivity import build_sensitivity_grids
from app.agents.specialized.valuation_specialist.state_valuation_specialist import (
    ValuationBuild,
    ValuationDataQuality,
    ValuationStance,
)
from app.agents.specialized.valuation_specialist.upside import upside_pct

if TYPE_CHECKING:  # runtime import would cycle through app.agents.specialized
    from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
    from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput

logger = logging.getLogger("finance_agent.agents.valuation_specialist")

STANCE_BAND_PCT = 10.0

# ==============================================================================
# Helpers
# ==============================================================================
def _stance(upside: Optional[float]) -> Optional[ValuationStance]:
    if upside is None:
        return None
    if upside > STANCE_BAND_PCT:
        return "Undervalued"
    if upside < -STANCE_BAND_PCT:
        return "Overvalued"
    return "Fairly Valued"


# ==============================================================================
# Main Assembly
# ==============================================================================
def build_valuation(
    ticker: str,
    fiscal_year: int,
    financial_audit: "FinancialAuditOutput",
    forecast: Optional["ForecastOutput"],
    market: MarketInputs,
    *,
    terminal_growth_rate: float,
    wacc_override: Optional[float] = None,
    valuation_date: Optional[date] = None,
    balance_sheet_date: Optional[str] = None,
) -> ValuationBuild:
    valuation_date = valuation_date or date.today()

    # 1. Cost of capital
    capital = estimate_cost_of_capital(gather_capital_inputs(financial_audit, market, wacc_override))
    wacc_audit, terminal_tax = capital.wacc_audit, capital.marginal_tax_rate
    wacc = wacc_audit.wacc

    # 2. Intrinsic value: base-case DCF rolled forward to the valuation date
    intrinsic = estimate_intrinsic_value(
        gather_intrinsic_value_inputs(financial_audit, forecast, market, valuation_date, balance_sheet_date),
        capital,
        terminal_growth_rate,
    )
    dcf, projected_fcfs, forecast_source = intrinsic.dcf, intrinsic.projected_fcfs, intrinsic.forecast_source
    grids = build_sensitivity_grids(dcf)
    warnings: List[str] = [*intrinsic.cash_flow_warnings, *capital.warnings, *intrinsic.warnings]
    fair = dcf.fair_value_per_share
    price = market.share_price if market.share_price and market.share_price > 0 else None
    upside = upside_pct(fair, price)

    # 3. Market-implied expectations (reverse DCF), same WACC / TV / bridge as the headline
    implied = solve_market_implied_expectations(forecast, intrinsic, capital, price)
    warnings.extend(implied.warnings)

    # 4. Scenario valuations (same WACC, growth and bridge; base reuses the headline DCF)
    scenarios = value_scenarios(forecast, intrinsic, capital, price)

    build = ValuationBuild(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        valuation_date=valuation_date.isoformat(),
        balance_sheet_date=intrinsic.balance_sheet_date.isoformat() if intrinsic.balance_sheet_date else None,
        roll_forward_years=dcf.roll_forward_years,
        wacc_audit=wacc_audit,
        terminal_growth_rate=terminal_growth_rate,
        terminal_tax_rate_pct=round(terminal_tax * 100.0, 2),
        discounting_convention=dcf.discounting_convention,
        terminal_method=dcf.terminal_method,
        terminal_fcf=dcf.terminal_fcf,
        ronic_used_pct=round(dcf.ronic * 100.0, 2) if dcf.ronic is not None else None,
        implied_terminal_ev_ebitda=dcf.implied_terminal_ev_ebitda,
        implied_terminal_ev_nopat=dcf.implied_terminal_ev_nopat,
        projected_fcfs=projected_fcfs,
        pv_explicit_fcfs=dcf.pv_explicit_fcfs,
        pv_terminal_value=dcf.pv_terminal_value,
        terminal_value_pct_of_ev=dcf.terminal_value_pct_of_ev,
        enterprise_value=dcf.enterprise_value,
        net_debt=dcf.net_debt,
        noncontrolling_interests=dcf.noncontrolling_interests,
        equity_value=dcf.equity_value,
        diluted_shares=dcf.diluted_shares,
        implied_fair_value_per_share=fair,
        current_share_price=price,
        upside_downside_pct=upside,
        valuation_stance=_stance(upside),
        implied_ev_ebitda=intrinsic.implied_ev_ebitda,
        scenario_valuations=scenarios.scenario_valuations,
        probability_weighted_fair_value=scenarios.probability_weighted_fair_value,
        bear_fair_value=scenarios.bear_fair_value,
        bull_fair_value=scenarios.bull_fair_value,
        market_implied_growth_pct=implied.market_implied_growth_pct,
        modeled_revenue_cagr_pct=implied.modeled_revenue_cagr_pct,
        reverse_dcf_horizon_years=implied.reverse_dcf_horizon_years,
        market_implied_margin_shift_pp=implied.market_implied_margin_shift_pp,
        modeled_final_margin_pct=implied.modeled_final_margin_pct,
        reverse_dcf_status=implied.reverse_dcf_status,
        reverse_dcf_summary=implied.reverse_dcf_summary,
        market_implied_wacc_pct=implied.market_implied_wacc_pct,
        sensitivity_matrix_markdown=grids.sensitivity_matrix_markdown,
        ronic_sensitivity_matrix_markdown=grids.ronic_sensitivity_matrix_markdown,
        equity_bridge_markdown=dcf.equity_bridge_markdown,
        scenario_table_markdown=scenarios.scenario_table_markdown,
        data_quality=ValuationDataQuality(
            forecast_source=forecast_source,
            market_data_source=market.market_data_source,
            share_price_source=market.share_price_source if price is not None else "unavailable",
            warnings=warnings,
        ),
    )
    logger.info(
        f"build_valuation {build.ticker} FY{fiscal_year}: WACC {wacc_audit.wacc_pct:.2f}% ({wacc_audit.wacc_source}), "
        f"base ${fair:.2f}, bear ${build.bear_fair_value}, bull ${build.bull_fair_value}, "
        f"weighted ${scenarios.probability_weighted_fair_value}, roll {intrinsic.terms.roll_forward_years:.2f}y, {len(warnings)} warnings"
    )
    return build
