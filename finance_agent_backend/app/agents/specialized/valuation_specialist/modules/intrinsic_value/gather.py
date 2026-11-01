"""Reads what the DCF values: the forecast cash flows (or the placeholder when there is none), the final forecast
year, the forecast's return diagnostics, and the audited bridge items."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional

from app.agents.specialized.valuation_specialist.context import MarketInputs
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.schema import IntrinsicValueInputs
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.thresholds import (
    NAIVE_FALLBACK_GROWTH,
    NAIVE_FALLBACK_YEARS,
)

if TYPE_CHECKING:
    from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
    from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def gather_intrinsic_value_inputs(
    audit: "FinancialAuditOutput",
    forecast: Optional["ForecastOutput"],
    market: MarketInputs,
    valuation_date: date,
    balance_sheet_date: Optional[str],
) -> IntrinsicValueInputs:
    bs = audit.balance_sheet
    if forecast is not None and forecast.projected_fcfs:
        projected_fcfs, forecast_source, cash_flow_warnings = list(forecast.projected_fcfs), "forecasting_analyst", ()
    else:
        latest = audit.multi_year_history[-1]
        projected_fcfs = [
            round(latest.free_cash_flow * (1.0 + NAIVE_FALLBACK_GROWTH) ** i, 2)
            for i in range(1, NAIVE_FALLBACK_YEARS + 1)
        ]
        forecast_source = "naive_5pct_fallback"
        cash_flow_warnings = ("No forecast available: UFCF is the last audited FCF grown 5% a year (placeholder).",)
    return IntrinsicValueInputs(
        projected_fcfs=projected_fcfs,
        forecast_source=forecast_source,
        cash_flow_warnings=cash_flow_warnings,
        final_year=forecast.forecast_schedule[-1] if forecast is not None and forecast.forecast_schedule else None,
        forecast_diagnostics=forecast.diagnostics if forecast is not None else None,
        net_debt=bs.net_debt,
        noncontrolling_interests=bs.noncontrolling_interests or 0.0,
        diluted_shares=bs.valuation_shares_outstanding,
        base_ebitda=audit.solvency_and_liquidity_ratios.ebitda,
        balance_sheet_date=_parse_date(balance_sheet_date),
        valuation_date=valuation_date,
        risk_free_rate=market.risk_free_rate,
    )
