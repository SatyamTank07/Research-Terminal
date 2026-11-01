"""Outside data the valuation needs: live market quotes and the 10Y Treasury (with offline fallbacks), the user's price / beta overrides, and the balance-sheet date."""

import logging
import math
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field

from app.config import settings
from app.database import SessionLocal
from app.models import ANNUAL_REPORT_FILING_TYPE, Document

logger = logging.getLogger("finance_agent.agents.valuation_specialist")

TREASURY_10Y_SYMBOL = "^TNX"


def _positive(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if not math.isnan(f) and f > 0 else None


def fetch_risk_free_rate() -> Dict[str, Any]:
    """10-year US Treasury yield from ^TNX (quoted in percent), else the configured default."""
    try:
        import yfinance as yf

        hist = yf.Ticker(TREASURY_10Y_SYMBOL).history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            yield_pct = _positive(hist["Close"].dropna().iloc[-1])
            if yield_pct is not None and 0.1 <= yield_pct <= 15.0:
                return {"risk_free_rate": round(yield_pct / 100.0, 5), "risk_free_rate_source": "us_10y_treasury_live"}
    except Exception as e:
        logger.debug(f"^TNX fetch failed: {e}")
    return {"risk_free_rate": settings.DEFAULT_RISK_FREE_RATE, "risk_free_rate_source": "house_default_config"}


def _offline(rf: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "share_price": None,
        "market_cap": None,
        "beta": None,
        "beta_source": "unavailable",
        "market_data_source": "fallback_offline",
        **rf,
    }


def fetch_market_context(ticker: str) -> Dict[str, Any]:
    """
    Fetches real-time equity market data from Yahoo Finance for the specified ticker.

    Returns:
        Dictionary with:
        - share_price: Latest trading price in $ (or None)
        - market_cap: Market capitalization in $ Millions (or None)
        - beta: Raw equity beta vs the S&P 500 (or None; never defaulted here)
        - beta_source: 'yfinance_raw' | 'unavailable'
        - risk_free_rate / risk_free_rate_source: live 10Y Treasury or configured default
        - market_data_source: 'yfinance_live' | 'fallback_offline'
    """
    rf = fetch_risk_free_rate()
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        return _offline(rf)

    try:
        import yfinance as yf

        t = yf.Ticker(clean_ticker)
        price = market_cap = beta = None

        # 1. Fast Info resolution (fastest, avoids full scraping overhead)
        try:
            fast_info = getattr(t, "fast_info", None)
            if fast_info:
                raw_price = _positive(getattr(fast_info, "last_price", None))
                if raw_price is not None:
                    price = round(raw_price, 2)
                raw_mkt_cap = _positive(getattr(fast_info, "market_cap", None))
                if raw_mkt_cap is not None:
                    market_cap = round(raw_mkt_cap / 1_000_000.0, 2)
        except Exception as e:
            logger.debug(f"fast_info extraction failed for {clean_ticker}: {e}")

        # 2. Historical close fallback if price was missing
        if price is None:
            try:
                hist = t.history(period="5d")
                if not hist.empty and "Close" in hist.columns:
                    last_close = _positive(hist["Close"].dropna().iloc[-1])
                    if last_close is not None:
                        price = round(last_close, 2)
            except Exception as e:
                logger.debug(f"history extraction failed for {clean_ticker}: {e}")

        # 3. Full info dictionary lookup for beta (and market_cap/price if still missing)
        try:
            info = getattr(t, "info", None) or {}
            raw_beta = _positive(info.get("beta"))
            if raw_beta is not None and raw_beta <= 5.0:
                beta = round(raw_beta, 2)
            if market_cap is None and _positive(info.get("marketCap")) is not None:
                market_cap = round(float(info["marketCap"]) / 1_000_000.0, 2)
            if price is None and _positive(info.get("regularMarketPrice")) is not None:
                price = round(float(info["regularMarketPrice"]), 2)
        except Exception as e:
            logger.debug(f"info extraction failed for {clean_ticker}: {e}")

        source = "yfinance_live" if (price is not None or market_cap is not None) else "fallback_offline"
        logger.info(
            f"[fetch_market_context] {clean_ticker}: price=${price}, market_cap=${market_cap}M, "
            f"beta={beta}, rf={rf['risk_free_rate']} ({rf['risk_free_rate_source']}) (source={source})"
        )
        return {
            "share_price": price,
            "market_cap": market_cap,
            "beta": beta,
            "beta_source": "yfinance_raw" if beta is not None else "unavailable",
            "market_data_source": source,
            **rf,
        }

    except Exception as e:
        logger.warning(f"Error fetching market data from yfinance for {clean_ticker}: {e}")
        return _offline(rf)


class MarketInputs(BaseModel):
    """Market data resolved here (live yfinance and/or user overrides) with provenance."""

    share_price: Optional[float] = None
    share_price_source: str = "unavailable"
    market_cap: Optional[float] = Field(None, description="$ Millions")
    beta: Optional[float] = None
    beta_source: str = Field("unavailable", description="'user_provided' | 'yfinance_raw' | 'unavailable'")
    risk_free_rate: float = settings.DEFAULT_RISK_FREE_RATE
    risk_free_rate_source: str = "house_default_config"
    market_data_source: str = "fallback_offline"

    @property
    def valid_share_price(self) -> Optional[float]:
        """The share price when it is positive, else None (then no upside and no stance)."""
        return self.share_price if self.share_price and self.share_price > 0 else None



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
