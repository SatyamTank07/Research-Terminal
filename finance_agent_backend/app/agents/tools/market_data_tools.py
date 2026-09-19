"""Market Data Context & Real-Time Pricing Resolver.

Provides live market capitalization, share price, and systematic equity beta
via yfinance with robust fallback guards for offline, rate-limited, or OTC tickers.
Standardizes all market capitalization figures to $ Millions for direct handoff
into the DCF and WACC calculation engines.
"""

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger("finance_agent.tools.market_data")


def fetch_market_context(ticker: str) -> Dict[str, Any]:
    """
    Fetches real-time equity market data from Yahoo Finance for the specified ticker.

    Args:
        ticker: Clean stock ticker symbol (e.g. 'AAPL', 'TSLA', 'NVDA').

    Returns:
        Dictionary with:
        - share_price: Latest trading price in $ (or None if unavailable)
        - market_cap: Market capitalization in $ Millions (or None if unavailable)
        - beta: Equity beta relative to S&P 500 (default 1.0 if unavailable)
        - market_data_source: Provenance string ('yfinance_live' | 'fallback_offline')
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        return {
            "share_price": None,
            "market_cap": None,
            "beta": 1.0,
            "market_data_source": "fallback_offline",
        }

    try:
        import yfinance as yf

        t = yf.Ticker(clean_ticker)

        # 1. Fast Info resolution (fastest, avoids full scraping overhead)
        price = None
        market_cap = None
        beta = None

        try:
            fast_info = getattr(t, "fast_info", None)
            if fast_info:
                raw_price = getattr(fast_info, "last_price", None)
                if raw_price is not None and not math.isnan(raw_price) and raw_price > 0:
                    price = round(float(raw_price), 2)

                raw_mkt_cap = getattr(fast_info, "market_cap", None)
                if raw_mkt_cap is not None and not math.isnan(raw_mkt_cap) and raw_mkt_cap > 0:
                    # Convert raw dollars to $ Millions
                    market_cap = round(float(raw_mkt_cap) / 1_000_000.0, 2)
        except Exception as e:
            logger.debug(f"fast_info extraction failed for {clean_ticker}: {e}")

        # 2. Historical close fallback if price was missing
        if price is None:
            try:
                hist = t.history(period="5d")
                if not hist.empty and "Close" in hist.columns:
                    last_close = float(hist["Close"].dropna().iloc[-1])
                    if last_close > 0:
                        price = round(last_close, 2)
            except Exception as e:
                logger.debug(f"history extraction failed for {clean_ticker}: {e}")

        # 3. Full info dictionary lookup for beta (and market_cap/price if still missing)
        try:
            info = getattr(t, "info", None) or {}
            raw_beta = info.get("beta")
            if raw_beta is not None and not math.isnan(raw_beta) and 0.0 < float(raw_beta) <= 5.0:
                beta = round(float(raw_beta), 2)

            if market_cap is None and info.get("marketCap"):
                raw_mc = float(info["marketCap"])
                if raw_mc > 0:
                    market_cap = round(raw_mc / 1_000_000.0, 2)

            if price is None and info.get("regularMarketPrice"):
                raw_p = float(info["regularMarketPrice"])
                if raw_p > 0:
                    price = round(raw_p, 2)
        except Exception as e:
            logger.debug(f"info extraction failed for {clean_ticker}: {e}")

        # If market_cap is still None but we have price and diluted shares estimate from yfinance
        if beta is None or beta <= 0:
            beta = 1.0

        source = "yfinance_live" if (price is not None or market_cap is not None) else "fallback_offline"

        logger.info(
            f"[fetch_market_context] {clean_ticker}: price=${price}, "
            f"market_cap=${market_cap}M, beta={beta} (source={source})"
        )

        return {
            "share_price": price,
            "market_cap": market_cap,
            "beta": beta,
            "market_data_source": source,
        }

    except Exception as e:
        logger.warning(f"Error fetching market data from yfinance for {clean_ticker}: {e}")
        return {
            "share_price": None,
            "market_cap": None,
            "beta": 1.0,
            "market_data_source": "fallback_offline",
        }
