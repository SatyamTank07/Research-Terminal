"""Assembles the module results into ValuationBuild. Warnings keep today's order: placeholder cash flows,
cost of capital, intrinsic value, reverse DCF."""

from datetime import date
from typing import Optional

from app.agents.specialized.valuation_specialist.context import MarketInputs
from app.agents.specialized.valuation_specialist.modules.cost_of_capital import CostOfCapital
from app.agents.specialized.valuation_specialist.modules.intrinsic_value import IntrinsicValue
from app.agents.specialized.valuation_specialist.modules.market_implied_expectations import MarketImpliedExpectations
from app.agents.specialized.valuation_specialist.modules.scenario_values import ScenarioValues
from app.agents.specialized.valuation_specialist.modules.sensitivity import Sensitivity
from app.agents.specialized.valuation_specialist.modules.valuation_stance import ValuationStanceResult
from app.agents.specialized.valuation_specialist.state_valuation_specialist import ValuationBuild, ValuationDataQuality


def assemble_valuation(
    *,
    ticker: str,
    fiscal_year: int,
    valuation_date: date,
    terminal_growth_rate: float,
    market: MarketInputs,
    price: Optional[float],
    capital: CostOfCapital,
    intrinsic: IntrinsicValue,
    scenarios: ScenarioValues,
    implied: MarketImpliedExpectations,
    grids: Sensitivity,
    stance: ValuationStanceResult,
) -> ValuationBuild:
    dcf = intrinsic.dcf
    return ValuationBuild(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        valuation_date=valuation_date.isoformat(),
        balance_sheet_date=intrinsic.balance_sheet_date.isoformat() if intrinsic.balance_sheet_date else None,
        roll_forward_years=dcf.roll_forward_years,
        wacc_audit=capital.wacc_audit,
        terminal_growth_rate=terminal_growth_rate,
        terminal_tax_rate_pct=round(capital.marginal_tax_rate * 100.0, 2),
        discounting_convention=dcf.discounting_convention,
        terminal_method=dcf.terminal_method,
        terminal_fcf=dcf.terminal_fcf,
        ronic_used_pct=round(dcf.ronic * 100.0, 2) if dcf.ronic is not None else None,
        implied_terminal_ev_ebitda=dcf.implied_terminal_ev_ebitda,
        implied_terminal_ev_nopat=dcf.implied_terminal_ev_nopat,
        projected_fcfs=intrinsic.projected_fcfs,
        pv_explicit_fcfs=dcf.pv_explicit_fcfs,
        pv_terminal_value=dcf.pv_terminal_value,
        terminal_value_pct_of_ev=dcf.terminal_value_pct_of_ev,
        enterprise_value=dcf.enterprise_value,
        net_debt=dcf.net_debt,
        noncontrolling_interests=dcf.noncontrolling_interests,
        equity_value=dcf.equity_value,
        diluted_shares=dcf.diluted_shares,
        implied_fair_value_per_share=dcf.fair_value_per_share,
        current_share_price=price,
        upside_downside_pct=stance.upside_downside_pct,
        valuation_stance=stance.valuation_stance,
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
            forecast_source=intrinsic.forecast_source,
            market_data_source=market.market_data_source,
            share_price_source=market.share_price_source if price is not None else "unavailable",
            warnings=[*intrinsic.cash_flow_warnings, *capital.warnings, *intrinsic.warnings, *implied.warnings],
        ),
    )
