"""Tests for the valuation context: live ^TNX risk-free rate, raw beta provenance and offline fallbacks (yfinance mocked)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from app.agents.specialized.valuation_specialist.context import fetch_market_context, fetch_risk_free_rate
from app.config import settings


def _ticker_factory(tnx_close=None, price=230.0, market_cap=3.45e12, beta=1.2, fail=False):
    def make(symbol):
        if fail:
            raise ConnectionError("offline")
        t = MagicMock()
        if symbol == "^TNX":
            t.history.return_value = pd.DataFrame({"Close": [tnx_close]}) if tnx_close is not None else pd.DataFrame()
        else:
            t.fast_info = SimpleNamespace(last_price=price, market_cap=market_cap)
            t.info = {"beta": beta}
        return t
    return make


class TestMarketData(unittest.TestCase):
    def test_live_treasury_yield_is_converted_to_decimal(self):
        with patch("yfinance.Ticker", side_effect=_ticker_factory(tnx_close=4.37)):
            rf = fetch_risk_free_rate()
        self.assertEqual(rf, {"risk_free_rate": 0.0437, "risk_free_rate_source": "us_10y_treasury_live"})

    def test_missing_or_implausible_yield_uses_config_default(self):
        for close in (None, 42.0):
            with patch("yfinance.Ticker", side_effect=_ticker_factory(tnx_close=close)):
                rf = fetch_risk_free_rate()
            self.assertEqual(rf["risk_free_rate"], settings.DEFAULT_RISK_FREE_RATE)
            self.assertEqual(rf["risk_free_rate_source"], "house_default_config")

    def test_context_reports_raw_beta_and_sources(self):
        with patch("yfinance.Ticker", side_effect=_ticker_factory(tnx_close=4.1)):
            ctx = fetch_market_context("aapl")
        self.assertEqual(ctx["share_price"], 230.0)
        self.assertEqual(ctx["market_cap"], 3_450_000.0)
        self.assertEqual((ctx["beta"], ctx["beta_source"]), (1.2, "yfinance_raw"))
        self.assertEqual(ctx["risk_free_rate"], 0.041)
        self.assertEqual(ctx["market_data_source"], "yfinance_live")

    def test_missing_beta_is_not_defaulted(self):
        with patch("yfinance.Ticker", side_effect=_ticker_factory(tnx_close=4.1, beta=None)):
            ctx = fetch_market_context("AAPL")
        self.assertIsNone(ctx["beta"])
        self.assertEqual(ctx["beta_source"], "unavailable")

    def test_offline_fallback(self):
        with patch("yfinance.Ticker", side_effect=_ticker_factory(fail=True)):
            ctx = fetch_market_context("AAPL")
        self.assertIsNone(ctx["share_price"])
        self.assertIsNone(ctx["beta"])
        self.assertEqual(ctx["market_data_source"], "fallback_offline")
        self.assertEqual(ctx["risk_free_rate_source"], "house_default_config")


if __name__ == "__main__":
    unittest.main()
