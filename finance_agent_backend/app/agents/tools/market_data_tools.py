"""Market Data Context & Real-Time Pricing Resolver.

Provides live market capitalization, share price, raw equity beta and the 10-year
US Treasury yield (risk-free rate) via yfinance with robust fallback guards for offline,
rate-limited, or OTC tickers. Every value carries a provenance tag so the valuation can
disclose when it ran on defaults. Market capitalization is standardized to $ Millions.
"""

import logging
import math
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger("finance_agent.tools.market_data")

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
