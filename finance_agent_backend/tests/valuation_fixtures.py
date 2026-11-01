"""Code-built valuation artifacts shared by the valuation, synthesizer and pipeline tests (no LLM)."""

from datetime import date
from typing import Optional

from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.specialized.valuation_specialist.agent_valuation_specialist import build_valuation
from app.agents.specialized.valuation_specialist.context import MarketInputs
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput
from app.agents.specialized.valuation_specialist.write import summarize_from_template
from tests.forecast_fixtures import make_aapl_audit, make_forecast_output

VALUATION_DATE = date(2026, 10, 11)
AAPL_BALANCE_SHEET_DATE = "2025-09-27"


def make_market(
    share_price: Optional[float] = 235.0,
    beta: Optional[float] = 1.10,
    beta_source: str = "yfinance_raw",
    risk_free_rate: float = 0.045,
    live_rf: bool = True,
) -> MarketInputs:
    return MarketInputs(
        share_price=share_price,
        share_price_source="yfinance_live" if share_price is not None else "unavailable",
        market_cap=None,
        beta=beta,
        beta_source=beta_source if beta is not None else "unavailable",
        risk_free_rate=risk_free_rate,
        risk_free_rate_source="us_10y_treasury_live" if live_rf else "house_default_config",
        market_data_source="yfinance_live" if share_price is not None else "fallback_offline",
    )


def make_valuation_output(
    ticker: str = "AAPL",
    fiscal_year: int = 2025,
    financial_audit: Optional[FinancialAuditOutput] = None,
    forecast: Optional[ForecastOutput] = None,
    market: Optional[MarketInputs] = None,
    terminal_growth_rate: float = 0.025,
    wacc_override: Optional[float] = None,
    balance_sheet_date: Optional[str] = AAPL_BALANCE_SHEET_DATE,
) -> DCFValuationOutput:
    build = build_valuation(
        ticker,
        fiscal_year,
        financial_audit or make_aapl_audit(),
        forecast if forecast is not None else make_forecast_output(ticker, fiscal_year),
        market or make_market(),
        terminal_growth_rate=terminal_growth_rate,
        wacc_override=wacc_override,
        valuation_date=VALUATION_DATE,
        balance_sheet_date=balance_sheet_date,
    )
    return DCFValuationOutput(
        **build.model_dump(),
        valuation_summary=summarize_from_template(build),
        commentary_source="template",
    )
