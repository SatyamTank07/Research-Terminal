"""Outside data the valuation needs: market quotes (with the user's overrides) and the balance-sheet date."""

import logging
from typing import Any, Mapping, Optional

from app.agents.tools.market_data_tools import fetch_market_context
from app.agents.tools.valuation_model import MarketInputs
from app.config import settings
from app.database import SessionLocal
from app.models import ANNUAL_REPORT_FILING_TYPE, Document

logger = logging.getLogger("finance_agent.agents.valuation_specialist")


def balance_sheet_date(ticker: str, fiscal_year: int) -> Optional[str]:
    """Fiscal year-end (period_end_date) of the anchor 10-K, used to roll the DCF forward to today."""
    db = SessionLocal()
    try:
        row = (
            db.query(Document.period_end_date)
            .filter(
                Document.ticker == ticker.upper(),
                Document.fiscal_year == fiscal_year,
                Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
            )
            .first()
        )
        return row[0] if row and row[0] else None
    except Exception as e:
        logger.warning(f"Balance-sheet date lookup failed for {ticker} FY{fiscal_year}: {e}")
        return None
    finally:
        db.close()


def market_inputs(ticker: str, overrides: Mapping[str, Any]) -> MarketInputs:
    """Live price, market cap, raw beta and 10Y Treasury, with user price / beta taking precedence."""
    user_beta, user_share_price = overrides.get("beta"), overrides.get("share_price")
    market_data = fetch_market_context(ticker)
    live_price = market_data.get("share_price")
    return MarketInputs(
        share_price=user_share_price if user_share_price is not None else live_price,
        share_price_source=(
            "user_provided" if user_share_price is not None
            else ("yfinance_live" if live_price is not None else "unavailable")
        ),
        # A user price makes the live market cap inconsistent; weights then use price x valuation shares.
        market_cap=None if user_share_price is not None else market_data.get("market_cap"),
        beta=user_beta if user_beta is not None else market_data.get("beta"),
        beta_source="user_provided" if user_beta is not None else market_data.get("beta_source", "unavailable"),
        risk_free_rate=market_data.get("risk_free_rate", settings.DEFAULT_RISK_FREE_RATE),
        risk_free_rate_source=market_data.get("risk_free_rate_source", "house_default_config"),
        market_data_source=market_data.get("market_data_source", "fallback_offline"),
    )
