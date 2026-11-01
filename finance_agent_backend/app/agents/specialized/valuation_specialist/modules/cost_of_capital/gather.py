"""Reads the cost-of-capital inputs from the audited balance sheet, the market quotes and the house assumptions."""

from typing import TYPE_CHECKING, List, Optional

from app.agents.specialized.valuation_specialist.context import MarketInputs
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import CostOfCapitalInputs
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.thresholds import (
    DEFAULT_MARKET_BETA,
    MARGINAL_TAX_RATE_BOUNDS,
)
from app.config import settings

if TYPE_CHECKING:
    from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput


def gather_capital_inputs(
    audit: "FinancialAuditOutput", market: MarketInputs, wacc_override: Optional[float]
) -> CostOfCapitalInputs:
    warnings: List[str] = []
    bs = audit.balance_sheet
    latest = audit.multi_year_history[-1]
    # Explicit years keep the normalized effective rate; the perpetuity and the debt tax shield use the
    # marginal rate, since tax deferrals and credits behind a low effective rate do not last forever.
    floor, cap = MARGINAL_TAX_RATE_BOUNDS
    marginal_tax_rate = min(max(settings.DEFAULT_MARGINAL_TAX_RATE, floor), cap)

    beta, beta_source, blume = market.beta, market.beta_source, market.beta_source == "yfinance_raw"
    if beta is None:
        beta, beta_source, blume = DEFAULT_MARKET_BETA, "default_market_beta", False
        warnings.append("No beta available; CAPM uses a market beta of 1.0.")
    if market.risk_free_rate_source != "us_10y_treasury_live":
        warnings.append(f"Risk-free rate {market.risk_free_rate * 100:.2f}% is the configured default, not a live yield.")

    equity_value = market.market_cap
    if equity_value is None and market.share_price is not None:
        equity_value = round(market.share_price * bs.valuation_shares_outstanding, 2)
    equity_value_source = "market_value"
    if equity_value is None:
        if bs.stockholders_equity > 0:
            equity_value, equity_value_source = bs.stockholders_equity, "book_equity_fallback"
            warnings.append("No market price: WACC weights use book equity, which understates the equity weight.")
        else:
            warnings.append("No market price and non-positive book equity: WACC weighted 100% equity.")

    return CostOfCapitalInputs(
        beta=beta,
        beta_source=beta_source,
        apply_blume_adjustment=blume,
        total_debt=bs.total_debt,
        prior_total_debt=bs.prior_total_debt,
        interest_expense=latest.interest_expense,
        equity_value=equity_value,
        equity_value_source=equity_value_source,
        risk_free_rate=market.risk_free_rate,
        risk_free_rate_source=market.risk_free_rate_source,
        equity_risk_premium=settings.DEFAULT_EQUITY_RISK_PREMIUM,
        equity_risk_premium_source=settings.EQUITY_RISK_PREMIUM_SOURCE,
        marginal_tax_rate=marginal_tax_rate,
        wacc_override=wacc_override,
        warnings=tuple(warnings),
    )
