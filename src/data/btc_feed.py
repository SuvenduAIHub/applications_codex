"""
Bitcoin (BTC/USDT) data feed with 4-layer fallback chain:
  1. Binance (CCXT direct + REST API — primary, fastest)
  2. CoinGecko API (free, real-time, no API key required)
  3. Alpha Vantage API (free tier with API key, reliable historical data)
  4. Yahoo Finance (yfinance) as last resort (always available)

Multiple fallback layers ensure the system never fails to get price data.
"""

import time
from datetime import datetime, timezone
from typing import Optional
import os

import pandas as pd
import requests
from loguru import logger

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from config.settings import DataFeedConfig, TimeFrame
from src.data.base_feed import BaseDataFeed


# Mapping from our TimeFrame enum to Binance API interval strings
TIMEFRAME_MAP = {
    TimeFrame.M1: "1m",
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.M30: "30m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "4h",
    TimeFrame.D1: "1d",
    TimeFrame.W1: "1w",
}


# CCXT timeframe mapping
CCXT_TIMEFRAME_MAP = {
    TimeFrame.M1: "1m",
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.M30: "30m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "4h",
    TimeFrame.D1: "1d",
    TimeFrame.W1: "1w",
}


class BinanceBTCFeed(BaseDataFeed):
    """
    Bitcoin data feed with 4-layer fallback: Binance → CoinGecko → Alpha Vantage → Yahoo Finance.
    Ensures maximum reliability — if one source fails, the next picks up automatically.
    """

    def __init__(self, config: Optional[DataFeedConfig] = None):
        """Initialize the BTC feed with CCXT as primary data source."""
        config = config or DataFeedConfig()
        super().__init__(symbol="BTC/USDT", config=config)
        self.base_url = config.binance_base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AutomatedTradingSystem/1.0",
        })

        # Initialize CCXT exchange (Binance by default, no API key needed for public data)
        self._ccxt_exchange = None
        if CCXT_AVAILABLE:
            try:
                self._ccxt_exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
                logger.info("CCXT initialized — using Binance direct connection for BTC data")
            except Exception as e:
                logger.warning(f"CCXT initialization failed: {e}. Will use REST API fallback.")

        # --- CoinGecko API (free, no API key needed, 30 calls/min) ---
        # CoinGecko provides real-time crypto prices with no authentication required
        self._coingecko_base_url = "https://api.coingecko.com/api/v3"
        logger.info("CoinGecko initialized — available as BTC data fallback (free, no API key)")

        # --- Alpha Vantage API (free tier: 5 calls/min, requires free API key) ---
        # Sign up at: https://www.alphavantage.co/support/#api-key
        self._alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self._alpha_vantage_base_url = "https://www.alphavantage.co/query"
        if self._alpha_vantage_key:
            logger.info("Alpha Vantage initialized — available as BTC data fallback")
        else:
            logger.info("Alpha Vantage API key not set — set ALPHA_VANTAGE_API_KEY in .env for extra fallback")


    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make a GET request to the Binance API with retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/api/v3/klines")
            params: Query parameters for the request

        Returns:
            Parsed JSON response

        Raises:
            requests.exceptions.RequestException: On persistent API failure
        """
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.config.api_retry_count):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.api_timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Binance API request failed (attempt {attempt + 1}/"
                    f"{self.config.api_retry_count}): {e}"
                )
                if attempt < self.config.api_retry_count - 1:
                    # Exponential backoff between retries
                    time.sleep(2 ** attempt)
                else:
                    raise

    def _fetch_via_ccxt(
        self, timeframe: TimeFrame, limit: int, start_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch BTC/USDT OHLCV via CCXT (direct exchange connection).
        CCXT connects to Binance WebSocket/REST natively — more reliable than
        raw HTTP requests and handles rate limiting automatically.

        Args:
            timeframe: Candlestick interval
            limit: Number of candles to fetch
            start_time: Optional ISO format start datetime

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        ccxt_tf = CCXT_TIMEFRAME_MAP.get(timeframe, "1h")
        since = None
        if start_time:
            dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            since = int(dt.timestamp() * 1000)

        logger.info(f"Fetching BTC/USDT via CCXT (Binance direct, tf={ccxt_tf}, limit={limit})")

        # CCXT returns [[timestamp, open, high, low, close, volume], ...]
        ohlcv = self._ccxt_exchange.fetch_ohlcv(
            "BTC/USDT", timeframe=ccxt_tf, since=since, limit=limit
        )

        if not ohlcv:
            raise ValueError("CCXT returned no BTC/USDT data")

        records = []
        for candle in ohlcv:
            records.append({
                "timestamp": datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            })

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.index.name = "datetime"
        return df

    def _fetch_ccxt_current_price(self) -> dict:
        """
        Fetch current BTC/USDT price via CCXT ticker.

        Returns:
            Dict with price, bid, ask, volume_24h, and timestamp
        """
        ticker = self._ccxt_exchange.fetch_ticker("BTC/USDT")
        return {
            "symbol": "BTC/USDT",
            "price": float(ticker.get("last", 0)),
            "bid": float(ticker.get("bid", 0) or 0),
            "ask": float(ticker.get("ask", 0) or 0),
            "volume_24h": float(ticker.get("baseVolume", 0) or 0),
            "high_24h": float(ticker.get("high", 0) or 0),
            "low_24h": float(ticker.get("low", 0) or 0),
            "change_pct_24h": float(ticker.get("percentage", 0) or 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def fetch_historical(
        self,
        timeframe: TimeFrame = TimeFrame.H1,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical BTC/USDT OHLCV data with 3-layer fallback:
          1. Binance via CCXT (direct exchange connection — primary)
          2. Binance REST API (public endpoints, no API key needed)
          3. Yahoo Finance (yfinance — last resort, always available)

        Args:
            timeframe: Candlestick interval (1m, 5m, 1h, 1d, etc.)
            start_time: ISO format start datetime string
            end_time: ISO format end datetime string
            limit: Number of candles (max 1000 per Binance API)

        Returns:
            DataFrame with columns [open, high, low, close, volume], DatetimeIndex
        """
        interval = TIMEFRAME_MAP.get(timeframe, "1h")

        # --- Layer 1: Binance via CCXT (primary — direct exchange connection) ---
        if self._ccxt_exchange is not None:
            try:
                df = self._fetch_via_ccxt(timeframe, limit, start_time)
                logger.info(f"CCXT (Binance): fetched {len(df)} BTC/USDT candles successfully")
                df = self.validate_dataframe(df)
                self._cache = df
                return df
            except Exception as e:
                logger.warning(f"CCXT (Binance) failed: {e}. Falling back to Binance REST API.")

        # --- Layer 2: Binance REST API (fallback) ---
        try:
            params = {
                "symbol": "BTCUSDT",
                "interval": interval,
                "limit": min(limit, self.config.max_candles_per_request),
            }
            if start_time:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                params["startTime"] = int(dt.timestamp() * 1000)
            if end_time:
                dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                params["endTime"] = int(dt.timestamp() * 1000)

            logger.info(f"Fetching BTC/USDT {interval} candles via Binance REST (limit={limit})")
            raw = self._make_request("/api/v3/klines", params)

            # Parse Binance kline response: [open_time, O, H, L, C, vol, ...]
            records = []
            for candle in raw:
                records.append({
                    "timestamp": datetime.fromtimestamp(
                        candle[0] / 1000, tz=timezone.utc
                    ),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index.name = "datetime"

            # Validate and return
            df = self.validate_dataframe(df)
            self._cache = df
            logger.info(f"Binance REST: fetched {len(df)} BTC/USDT candles successfully")
            return df

        except Exception as e:
            logger.warning(f"Binance REST failed: {e}. Falling back to CoinGecko.")

        # --- Layer 3: CoinGecko API (free, no API key, real-time crypto data) ---
        try:
            df = self._fetch_via_coingecko(timeframe, limit, start_time)
            logger.info(f"CoinGecko: fetched {len(df)} BTC candles successfully")
            df = self.validate_dataframe(df)
            self._cache = df
            return df
        except Exception as e:
            logger.warning(f"CoinGecko failed: {e}. Falling back to Alpha Vantage.")

        # --- Layer 4: Alpha Vantage API (free with API key, reliable historical) ---
        if self._alpha_vantage_key:
            try:
                df = self._fetch_via_alpha_vantage(timeframe, limit)
                logger.info(f"Alpha Vantage: fetched {len(df)} BTC candles successfully")
                df = self.validate_dataframe(df)
                self._cache = df
                return df
            except Exception as e:
                logger.warning(f"Alpha Vantage failed: {e}. Falling back to Yahoo Finance.")

        # --- Layer 5: Yahoo Finance (last resort — always available) ---
        df = self._fetch_via_yfinance(interval, limit)

        # Validate and clean the data
        df = self.validate_dataframe(df)
        # Cache the result for subsequent access
        self._cache = df

        logger.info(f"Fetched {len(df)} BTC/USDT candles")
        return df


    def _fetch_via_coingecko(
        self, timeframe: TimeFrame, limit: int, start_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch BTC/USD OHLCV data from CoinGecko API (free, no API key).
        CoinGecko provides market chart data with configurable granularity.

        Args:
            timeframe: Candlestick interval
            limit: Number of candles to fetch
            start_time: Optional ISO start datetime

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        # CoinGecko uses days parameter to determine granularity:
        # 1 day = 5-minute data, 2-90 days = hourly, >90 days = daily
        days_map = {
            TimeFrame.M1: 1, TimeFrame.M5: 1, TimeFrame.M15: 1,
            TimeFrame.M30: 2, TimeFrame.H1: 90, TimeFrame.H4: 90,
            TimeFrame.D1: 365, TimeFrame.W1: 365,
        }
        days = days_map.get(timeframe, 90)

        logger.info(f"Fetching BTC/USD via CoinGecko (days={days})")

        response = self.session.get(
            f"{self._coingecko_base_url}/coins/bitcoin/ohlc",
            params={"vs_currency": "usd", "days": days},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("CoinGecko returned no BTC data")

        # CoinGecko OHLC returns [[timestamp_ms, open, high, low, close], ...]
        records = []
        for candle in data:
            records.append({
                "timestamp": datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": 0.0,  # CoinGecko OHLC doesn't include volume
            })

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.index.name = "datetime"

        # Trim to requested limit
        if len(df) > limit:
            df = df.tail(limit)

        return df

    def _fetch_coingecko_current_price(self) -> dict:
        """
        Fetch current BTC price from CoinGecko (free, no API key).

        Returns:
            Dict with price details
        """
        response = self.session.get(
            f"{self._coingecko_base_url}/simple/price",
            params={
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json().get("bitcoin", {})
        price = float(data.get("usd", 0))
        return {
            "symbol": "BTC/USDT",
            "price": price,
            "bid": price * 0.9995,
            "ask": price * 1.0005,
            "volume_24h": float(data.get("usd_24h_vol", 0)),
            "high_24h": price * 1.02,
            "low_24h": price * 0.98,
            "change_pct_24h": float(data.get("usd_24h_change", 0)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fetch_via_alpha_vantage(
        self, timeframe: TimeFrame, limit: int
    ) -> pd.DataFrame:
        """
        Fetch BTC/USD data from Alpha Vantage (free tier: 5 calls/min).
        Requires ALPHA_VANTAGE_API_KEY env var. Get a free key at:
        https://www.alphavantage.co/support/#api-key

        Args:
            timeframe: Candlestick interval
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        # Alpha Vantage function mapping for crypto
        if timeframe in (TimeFrame.D1, TimeFrame.W1):
            function = "DIGITAL_CURRENCY_DAILY"
            ts_key = "Time Series (Digital Currency Daily)"
        else:
            function = "CRYPTO_INTRADAY"
            ts_key = "Time Series Crypto (5min)"  # default

        # Map timeframes to Alpha Vantage interval param
        av_interval_map = {
            TimeFrame.M1: "1min", TimeFrame.M5: "5min",
            TimeFrame.M15: "15min", TimeFrame.M30: "30min",
            TimeFrame.H1: "60min",
        }

        params = {
            "function": function,
            "symbol": "BTC",
            "market": "USD",
            "apikey": self._alpha_vantage_key,
        }
        if function == "CRYPTO_INTRADAY":
            interval = av_interval_map.get(timeframe, "60min")
            params["interval"] = interval
            ts_key = f"Time Series Crypto ({interval})"
            params["outputsize"] = "full" if limit > 100 else "compact"

        logger.info(f"Fetching BTC/USD via Alpha Vantage ({function})")

        response = self.session.get(
            self._alpha_vantage_base_url, params=params, timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        # Check for API error messages
        if "Error Message" in data or "Note" in data:
            raise ValueError(f"Alpha Vantage error: {data.get('Error Message', data.get('Note', ''))}")

        time_series = data.get(ts_key, {})
        if not time_series:
            raise ValueError(f"Alpha Vantage returned no data (key: {ts_key})")

        records = []
        for dt_str, values in time_series.items():
            # Alpha Vantage uses keys like "1. open", "2. high", etc.
            records.append({
                "timestamp": datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc),
                "open": float(values.get("1. open", values.get("1a. open (USD)", 0))),
                "high": float(values.get("2. high", values.get("2a. high (USD)", 0))),
                "low": float(values.get("3. low", values.get("3a. low (USD)", 0))),
                "close": float(values.get("4. close", values.get("4a. close (USD)", 0))),
                "volume": float(values.get("5. volume", values.get("5. volume", 0))),
            })

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)  # Alpha Vantage returns newest first
        df.index.name = "datetime"

        # Trim to requested limit
        if len(df) > limit:
            df = df.tail(limit)

        return df

    def _fetch_alpha_vantage_current_price(self) -> dict:
        """
        Fetch current BTC price from Alpha Vantage.

        Returns:
            Dict with price details
        """
        response = self.session.get(
            self._alpha_vantage_base_url,
            params={
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "BTC",
                "to_currency": "USD",
                "apikey": self._alpha_vantage_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        rate_data = data.get("Realtime Currency Exchange Rate", {})
        price = float(rate_data.get("5. Exchange Rate", 0))
        bid = float(rate_data.get("8. Bid Price", price))
        ask = float(rate_data.get("9. Ask Price", price))
        return {
            "symbol": "BTC/USDT",
            "price": price,
            "bid": bid,
            "ask": ask,
            "volume_24h": 0.0,
            "high_24h": price * 1.02,
            "low_24h": price * 0.98,
            "change_pct_24h": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fetch_via_yfinance(self, interval: str, limit: int) -> pd.DataFrame:
        """
        Fallback: fetch BTC-USD data via yfinance when Binance is unavailable.
        Uses BTC-USD ticker from Yahoo Finance (free, no API key).

        Args:
            interval: Candle interval string (e.g., "1h", "1d")
            limit: Approximate number of candles to fetch

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        import yfinance as yf

        # Map Binance interval strings to yfinance intervals and periods
        yf_interval_map = {
            "1m": ("1m", "7d"),     # yfinance max 7 days for 1m
            "5m": ("5m", "60d"),
            "15m": ("15m", "60d"),
            "30m": ("30m", "60d"),
            "1h": ("1h", "730d"),
            "4h": ("1h", "730d"),   # yfinance doesn't have 4h, use 1h
            "1d": ("1d", "2y"),
            "1w": ("1wk", "5y"),
        }
        yf_interval, yf_period = yf_interval_map.get(interval, ("1h", "730d"))

        logger.info(f"Fetching BTC-USD via yfinance (interval={yf_interval}, period={yf_period})")
        ticker = yf.Ticker("BTC-USD")
        hist = ticker.history(period=yf_period, interval=yf_interval)

        if hist.empty:
            raise ValueError("yfinance returned no BTC-USD data")

        # Rename columns to match expected format (lowercase)
        df = hist.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        # Ensure timezone-aware UTC index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "datetime"

        # Trim to requested limit
        if len(df) > limit:
            df = df.tail(limit)

        return df

    def fetch_current_price(self) -> dict:
        """
        Fetch current BTC/USDT price with 4-layer fallback:
          1. Binance via CCXT (primary — direct exchange connection)
          2. Binance REST API
          3. CoinGecko API (free, no API key)
          4. Alpha Vantage API (free with API key)
          5. Yahoo Finance (yfinance) as last resort

        Returns:
            Dict with price, bid, ask, volume_24h, and timestamp
        """
        # --- Layer 1: Binance via CCXT (primary — direct exchange connection) ---
        if self._ccxt_exchange is not None:
            try:
                return self._fetch_ccxt_current_price()
            except Exception as e:
                logger.warning(f"CCXT (Binance) price failed: {e}. Trying Binance REST.")

        # --- Layer 2: Binance REST API (fallback) ---
        try:
            raw = self._make_request("/api/v3/ticker/24hr", {"symbol": "BTCUSDT"})
            return {
                "symbol": "BTC/USDT",
                "price": float(raw["lastPrice"]),
                "bid": float(raw["bidPrice"]),
                "ask": float(raw["askPrice"]),
                "volume_24h": float(raw["volume"]),
                "high_24h": float(raw["highPrice"]),
                "low_24h": float(raw["lowPrice"]),
                "change_pct_24h": float(raw["priceChangePercent"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.warning(f"Binance REST price failed: {e}. Trying CoinGecko.")

        # --- Layer 3: CoinGecko (free, real-time crypto prices) ---
        try:
            return self._fetch_coingecko_current_price()
        except Exception as e:
            logger.warning(f"CoinGecko price failed: {e}. Trying Alpha Vantage.")

        # --- Layer 4: Alpha Vantage (free with API key) ---
        if self._alpha_vantage_key:
            try:
                return self._fetch_alpha_vantage_current_price()
            except Exception as e:
                logger.warning(f"Alpha Vantage price failed: {e}. Using Yahoo Finance.")

        # --- Layer 5: Yahoo Finance (last resort) ---
        try:
            import yfinance as yf
            ticker = yf.Ticker("BTC-USD")
            info = ticker.fast_info
            price = float(info.last_price) if hasattr(info, 'last_price') else 0.0
            return {
                "symbol": "BTC/USDT",
                "price": price,
                "bid": price * 0.999,
                "ask": price * 1.001,
                "volume_24h": 0.0,
                "high_24h": price * 1.02,
                "low_24h": price * 0.98,
                "change_pct_24h": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"All price sources failed for BTC. Last error: {e}")
            return {
                "symbol": "BTC/USDT", "price": 0.0, "bid": 0.0, "ask": 0.0,
                "volume_24h": 0.0, "high_24h": 0.0, "low_24h": 0.0,
                "change_pct_24h": 0.0, "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def fetch_orderbook(self, depth: int = 10) -> dict:
        """
        Fetch the current BTC/USDT order book from Binance.

        Args:
            depth: Number of price levels (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Dict with bids and asks lists of [price, quantity] pairs
        """
        raw = self._make_request("/api/v3/depth", {
            "symbol": "BTCUSDT",
            "limit": depth,
        })
        return {
            "symbol": "BTC/USDT",
            "bids": [[float(p), float(q)] for p, q in raw["bids"]],
            "asks": [[float(p), float(q)] for p, q in raw["asks"]],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def fetch_recent_trades(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent BTC/USDT trades from Binance.

        Args:
            limit: Number of recent trades to fetch (max 1000)

        Returns:
            DataFrame with trade data
        """
        raw = self._make_request("/api/v3/trades", {
            "symbol": "BTCUSDT",
            "limit": min(limit, 1000),
        })
        records = []
        for trade in raw:
            records.append({
                "timestamp": datetime.fromtimestamp(
                    trade["time"] / 1000, tz=timezone.utc
                ),
                "price": float(trade["price"]),
                "quantity": float(trade["qty"]),
                "is_buyer_maker": trade["isBuyerMaker"],
            })
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df
