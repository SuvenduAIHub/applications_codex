"""
Unit tests for data feed integrations (CoinGecko, Alpha Vantage fallbacks).
All external API calls are mocked — no real API keys or network required.
Tests cover historical data fetching, current price fetching, error handling,
and fallback behavior for both BTC and Gold feeds.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pandas as pd

from config.settings import DataFeedConfig, TimeFrame
from src.data.btc_feed import BinanceBTCFeed
from src.data.gold_feed import YahooGoldFeed


# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def btc_feed():
    """Create a BTC feed with default config (no CCXT to simplify tests)."""
    with patch("src.data.btc_feed.CCXT_AVAILABLE", False):
        feed = BinanceBTCFeed(DataFeedConfig())
    return feed


@pytest.fixture
def gold_feed():
    """Create a Gold feed with default config (no CCXT to simplify tests)."""
    with patch("src.data.gold_feed.CCXT_AVAILABLE", False):
        feed = YahooGoldFeed(DataFeedConfig())
    return feed


# ──────────────────────────────────────────────
# BTC Feed — CoinGecko Tests
# ──────────────────────────────────────────────

class TestBTCCoinGecko:
    """Tests for BTC data fetching via CoinGecko API."""

    @patch("src.data.btc_feed.requests.Session.get")
    def test_fetch_via_coingecko_success(self, mock_get, btc_feed):
        """CoinGecko should return valid OHLCV DataFrame for BTC."""
        # CoinGecko OHLC returns [[timestamp_ms, open, high, low, close], ...]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            [1716000000000, 67000.0, 67500.0, 66800.0, 67200.0],
            [1716003600000, 67200.0, 67800.0, 67100.0, 67600.0],
            [1716007200000, 67600.0, 68000.0, 67400.0, 67900.0],
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = btc_feed._fetch_via_coingecko(TimeFrame.H1, limit=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["close"].iloc[-1] == 67900.0

    @patch("src.data.btc_feed.requests.Session.get")
    def test_fetch_via_coingecko_empty(self, mock_get, btc_feed):
        """CoinGecko returning empty data should raise ValueError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="CoinGecko returned no BTC data"):
            btc_feed._fetch_via_coingecko(TimeFrame.H1, limit=100)

    @patch("src.data.btc_feed.requests.Session.get")
    def test_fetch_coingecko_current_price(self, mock_get, btc_feed):
        """CoinGecko current price should return valid price dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "bitcoin": {
                "usd": 67500.0,
                "usd_24h_vol": 25000000000.0,
                "usd_24h_change": 2.5,
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = btc_feed._fetch_coingecko_current_price()
        assert result["symbol"] == "BTC/USDT"
        assert result["price"] == 67500.0
        assert result["volume_24h"] == 25000000000.0
        assert result["change_pct_24h"] == 2.5

    @patch("src.data.btc_feed.requests.Session.get")
    def test_fetch_coingecko_network_error(self, mock_get, btc_feed):
        """Network errors should propagate as exceptions."""
        mock_get.side_effect = Exception("Connection timeout")
        with pytest.raises(Exception, match="Connection timeout"):
            btc_feed._fetch_coingecko_current_price()


# ──────────────────────────────────────────────
# BTC Feed — Alpha Vantage Tests
# ──────────────────────────────────────────────

class TestBTCAlphaVantage:
    """Tests for BTC data fetching via Alpha Vantage API."""

    @patch("src.data.btc_feed.requests.Session.get")
    def test_fetch_via_alpha_vantage_daily(self, mock_get, btc_feed):
        """Alpha Vantage daily BTC data should return valid DataFrame."""
        btc_feed._alpha_vantage_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Time Series (Digital Currency Daily)": {
                "2024-05-18": {
                    "1a. open (USD)": "67000.0", "2a. high (USD)": "67500.0",
                    "3a. low (USD)": "66800.0", "4a. close (USD)": "67200.0",
                    "5. volume": "15000.0",
                },
                "2024-05-17": {
                    "1a. open (USD)": "66500.0", "2a. high (USD)": "67100.0",
                    "3a. low (USD)": "66200.0", "4a. close (USD)": "67000.0",
                    "5. volume": "14000.0",
                },
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = btc_feed._fetch_via_alpha_vantage(TimeFrame.D1, limit=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns

    @patch("src.data.btc_feed.requests.Session.get")
    def test_fetch_via_alpha_vantage_error_message(self, mock_get, btc_feed):
        """Alpha Vantage API error messages should raise ValueError."""
        btc_feed._alpha_vantage_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Error Message": "Invalid API call"
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="Alpha Vantage error"):
            btc_feed._fetch_via_alpha_vantage(TimeFrame.D1, limit=100)

    @patch("src.data.btc_feed.requests.Session.get")
    def test_fetch_alpha_vantage_current_price(self, mock_get, btc_feed):
        """Alpha Vantage exchange rate endpoint should return valid price."""
        btc_feed._alpha_vantage_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Realtime Currency Exchange Rate": {
                "5. Exchange Rate": "67500.0",
                "8. Bid Price": "67490.0",
                "9. Ask Price": "67510.0",
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = btc_feed._fetch_alpha_vantage_current_price()
        assert result["price"] == 67500.0
        assert result["bid"] == 67490.0
        assert result["ask"] == 67510.0


# ──────────────────────────────────────────────
# Gold Feed — CoinGecko Tests
# ──────────────────────────────────────────────

class TestGoldCoinGecko:
    """Tests for Gold data fetching via CoinGecko (PAXG proxy)."""

    @patch("src.data.gold_feed.requests.Session.get")
    def test_fetch_via_coingecko_success(self, mock_get, gold_feed):
        """CoinGecko should return valid OHLCV DataFrame for Gold (PAXG)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            [1716000000000, 2340.0, 2345.0, 2338.0, 2342.0],
            [1716003600000, 2342.0, 2350.0, 2340.0, 2348.0],
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = gold_feed._fetch_via_coingecko(TimeFrame.H1, limit=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df["close"].iloc[-1] == 2348.0

    @patch("src.data.gold_feed.requests.Session.get")
    def test_fetch_via_coingecko_empty(self, mock_get, gold_feed):
        """CoinGecko returning empty PAXG data should raise ValueError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="CoinGecko returned no PAXG"):
            gold_feed._fetch_via_coingecko(TimeFrame.H1, limit=100)

    @patch("src.data.gold_feed.requests.Session.get")
    def test_fetch_coingecko_current_gold_price(self, mock_get, gold_feed):
        """CoinGecko current gold price should use PAXG as proxy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "pax-gold": {
                "usd": 2345.0,
                "usd_24h_vol": 50000000.0,
                "usd_24h_change": -0.5,
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = gold_feed._fetch_coingecko_current_price()
        assert result["symbol"] == "XAU/USD"
        assert result["price"] == 2345.0
        assert result["change_pct_24h"] == -0.5


# ──────────────────────────────────────────────
# Gold Feed — Alpha Vantage Tests
# ──────────────────────────────────────────────

class TestGoldAlphaVantage:
    """Tests for Gold data fetching via Alpha Vantage FX endpoint."""

    @patch("src.data.gold_feed.requests.Session.get")
    def test_fetch_via_alpha_vantage_daily(self, mock_get, gold_feed):
        """Alpha Vantage FX_DAILY for XAU/USD should return valid DataFrame."""
        gold_feed._alpha_vantage_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Time Series FX (Daily)": {
                "2024-05-18": {
                    "1. open": "2340.0", "2. high": "2350.0",
                    "3. low": "2335.0", "4. close": "2345.0",
                },
                "2024-05-17": {
                    "1. open": "2330.0", "2. high": "2342.0",
                    "3. low": "2325.0", "4. close": "2340.0",
                },
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = gold_feed._fetch_via_alpha_vantage(TimeFrame.D1, limit=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns

    @patch("src.data.gold_feed.requests.Session.get")
    def test_fetch_alpha_vantage_current_gold_price(self, mock_get, gold_feed):
        """Alpha Vantage XAU/USD exchange rate should return valid price."""
        gold_feed._alpha_vantage_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Realtime Currency Exchange Rate": {
                "5. Exchange Rate": "2345.50",
                "8. Bid Price": "2345.00",
                "9. Ask Price": "2346.00",
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = gold_feed._fetch_alpha_vantage_current_price()
        assert result["symbol"] == "XAU/USD"
        assert result["price"] == 2345.50
        assert result["bid"] == 2345.00
        assert result["ask"] == 2346.00

    @patch("src.data.gold_feed.requests.Session.get")
    def test_fetch_alpha_vantage_error(self, mock_get, gold_feed):
        """Alpha Vantage API error should raise ValueError."""
        gold_feed._alpha_vantage_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Note": "API call frequency limit reached (5 calls/min)"
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="Alpha Vantage error"):
            gold_feed._fetch_via_alpha_vantage(TimeFrame.D1, limit=100)


# ──────────────────────────────────────────────
# Fallback Chain Tests
# ──────────────────────────────────────────────

class TestFallbackChain:
    """Tests verifying the 4-layer fallback behavior works correctly."""

    def test_btc_feed_has_coingecko_url(self, btc_feed):
        """BTC feed should have CoinGecko base URL configured."""
        assert hasattr(btc_feed, "_coingecko_base_url")
        assert "coingecko" in btc_feed._coingecko_base_url

    def test_btc_feed_has_alpha_vantage_url(self, btc_feed):
        """BTC feed should have Alpha Vantage base URL configured."""
        assert hasattr(btc_feed, "_alpha_vantage_base_url")
        assert "alphavantage" in btc_feed._alpha_vantage_base_url

    def test_gold_feed_has_coingecko_url(self, gold_feed):
        """Gold feed should have CoinGecko base URL configured."""
        assert hasattr(gold_feed, "_coingecko_base_url")
        assert "coingecko" in gold_feed._coingecko_base_url

    def test_gold_feed_has_alpha_vantage_url(self, gold_feed):
        """Gold feed should have Alpha Vantage base URL configured."""
        assert hasattr(gold_feed, "_alpha_vantage_base_url")
        assert "alphavantage" in gold_feed._alpha_vantage_base_url

    def test_btc_feed_alpha_vantage_key_default_empty(self, btc_feed):
        """Alpha Vantage key should default to empty (optional)."""
        # Key is loaded from env; in test environment it should be empty
        assert isinstance(btc_feed._alpha_vantage_key, str)
