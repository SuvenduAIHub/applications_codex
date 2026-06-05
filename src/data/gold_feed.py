"""
Gold (XAU/USD) data feed with 4-layer fallback chain:
  1. CoinGecko API (free, real-time gold via PAXG proxy, no API key)
  2. Alpha Vantage API (free tier with API key, FX_DAILY for XAU/USD)
  3. CCXT (direct exchange connection — PAXG/USDT gold-backed token)
  4. Yahoo Finance (GC=F gold futures — last resort, always available)

Multiple fallback layers ensure gold price data is always available.
CoinGecko uses PAXG (PAX Gold) as a 24/7 gold price proxy.
Yahoo Finance uses gold futures (GC=F) as the last resort.
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

# CCXT timeframe mapping for gold pairs
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


# Mapping from our TimeFrame enum to Yahoo Finance interval strings
TIMEFRAME_MAP = {
    TimeFrame.M1: "1m",
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.M30: "30m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "1h",      # Yahoo doesn't support 4h, so we fetch 1h and resample
    TimeFrame.D1: "1d",
    TimeFrame.W1: "1wk",
}

# Yahoo Finance range strings corresponding to timeframes
# (defines max lookback per interval)
RANGE_MAP = {
    TimeFrame.M1: "7d",
    TimeFrame.M5: "60d",
    TimeFrame.M15: "60d",
    TimeFrame.M30: "60d",
    TimeFrame.H1: "730d",
    TimeFrame.H4: "730d",
    TimeFrame.D1: "10y",
    TimeFrame.W1: "10y",
}


# OANDA timeframe mapping (granularity strings for their REST API)
OANDA_TIMEFRAME_MAP = {
    TimeFrame.M1: "M1",
    TimeFrame.M5: "M5",
    TimeFrame.M15: "M15",
    TimeFrame.M30: "M30",
    TimeFrame.H1: "H1",
    TimeFrame.H4: "H4",
    TimeFrame.D1: "D",
    TimeFrame.W1: "W",
}


class YahooGoldFeed(BaseDataFeed):
    """
    Gold (XAU/USD) data feed with 4-layer fallback: CoinGecko → Alpha Vantage → CCXT → Yahoo Finance.
    Ensures maximum reliability — if one source fails, the next picks up automatically.
    """

    def __init__(self, config: Optional[DataFeedConfig] = None):
        """Initialize the Gold feed with OANDA as primary, CCXT and Yahoo as fallbacks."""
        config = config or DataFeedConfig()
        super().__init__(symbol="XAU/USD", config=config)
        self.base_url = config.gold_api_url
        # Yahoo Finance symbol for gold futures
        self.yahoo_symbol = "GC=F"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (AutomatedTradingSystem/1.0)",
        })

        # OANDA API configuration — uses practice/demo endpoint by default
        # Set OANDA_API_KEY in .env for authenticated access (higher rate limits)
        # Free OANDA practice accounts available at https://www.oanda.com/register
        import os
        self._oanda_api_key = os.environ.get("OANDA_API_KEY", "")
        self._oanda_account_type = os.environ.get("OANDA_ACCOUNT_TYPE", "practice")
        if self._oanda_account_type == "live":
            self._oanda_base_url = "https://api-fxtrade.oanda.com"
        else:
            self._oanda_base_url = "https://api-fxpractice.oanda.com"
        self._oanda_instrument = "XAU_USD"  # OANDA uses underscore format

        if self._oanda_api_key:
            logger.info(f"OANDA initialized — using {self._oanda_account_type} API for Gold data")
        else:
            logger.info("OANDA API key not set — will try CCXT/Yahoo for Gold data. "
                        "Set OANDA_API_KEY in .env for professional gold data.")

        # Initialize CCXT for gold-proxy pairs (PAXG/USDT on Binance)
        # PAXG is PAX Gold — each token is backed by 1 oz of physical gold
        self._ccxt_exchange = None
        self._ccxt_gold_symbol = None
        if CCXT_AVAILABLE:
            try:
                self._ccxt_exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
                # PAXG/USDT tracks physical gold price very closely
                self._ccxt_gold_symbol = "PAXG/USDT"
                logger.info("CCXT initialized — using PAXG/USDT (gold-backed token) as Gold fallback")
            except Exception as e:
                logger.warning(f"CCXT init for Gold failed: {e}. Will use Yahoo Finance.")

        # --- CoinGecko API (free, no API key needed, gold via PAXG proxy) ---
        # CoinGecko tracks PAXG (PAX Gold) which is backed 1:1 by physical gold
        self._coingecko_base_url = "https://api.coingecko.com/api/v3"
        logger.info("CoinGecko initialized — available as Gold data fallback (free, no API key)")

        # --- Alpha Vantage API (free tier: 5 calls/min, FX_DAILY for XAU/USD) ---
        # Sign up for free key at: https://www.alphavantage.co/support/#api-key
        self._alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self._alpha_vantage_base_url = "https://www.alphavantage.co/query"
        if self._alpha_vantage_key:
            logger.info("Alpha Vantage initialized — available as Gold data fallback")
        else:
            logger.info("Alpha Vantage API key not set — set ALPHA_VANTAGE_API_KEY in .env for extra fallback")


    def _make_request(self, params: dict) -> dict:
        """
        Make a GET request to Yahoo Finance chart API with retry logic.

        Args:
            params: Query parameters for the chart endpoint

        Returns:
            Parsed JSON response

        Raises:
            requests.exceptions.RequestException: On persistent API failure
        """
        url = f"{self.base_url}/{self.yahoo_symbol}"
        for attempt in range(self.config.api_retry_count):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.api_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                # Yahoo wraps results inside chart.result[0]
                if "chart" in data and data["chart"]["result"]:
                    return data["chart"]["result"][0]
                raise ValueError("No data returned from Yahoo Finance")
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(
                    f"Yahoo Finance API failed (attempt {attempt + 1}/"
                    f"{self.config.api_retry_count}): {e}"
                )
                if attempt < self.config.api_retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def _fetch_via_oanda(
        self, timeframe: TimeFrame, limit: int, start_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch XAU/USD OHLCV data from OANDA REST API.
        OANDA is a professional forex/commodities broker with institutional-grade
        gold price data. Requires a free practice API key.

        Sign up at: https://www.oanda.com/register
        Then create an API token in your account settings.

        Args:
            timeframe: Candlestick interval
            limit: Number of candles to fetch (max 5000 per OANDA request)
            start_time: Optional ISO start datetime

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        granularity = OANDA_TIMEFRAME_MAP.get(timeframe, "H1")
        url = f"{self._oanda_base_url}/v3/instruments/{self._oanda_instrument}/candles"

        headers = {
            "Authorization": f"Bearer {self._oanda_api_key}",
            "Content-Type": "application/json",
        }

        params = {
            "granularity": granularity,
            "count": min(limit, 5000),  # OANDA max is 5000 per request
            "price": "M",  # Mid prices (average of bid/ask)
        }
        if start_time:
            params["from"] = start_time

        logger.info(f"Fetching XAU/USD via OANDA ({granularity}, limit={limit})")

        response = self.session.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        candles = data.get("candles", [])
        if not candles:
            raise ValueError("OANDA returned no XAU/USD candle data")

        records = []
        for candle in candles:
            if not candle.get("complete", False) and len(candles) > 1:
                continue  # Skip incomplete candles (except if it's the only one)
            mid = candle.get("mid", {})
            records.append({
                "timestamp": datetime.fromisoformat(
                    candle["time"].replace("Z", "+00:00").split(".")[0] + "+00:00"
                ),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": float(candle.get("volume", 0)),
            })

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.index.name = "datetime"
        return df

    def _fetch_oanda_current_price(self) -> dict:
        """
        Fetch current XAU/USD price from OANDA pricing endpoint.

        Returns:
            Dict with price, bid, ask, and timestamp
        """
        url = f"{self._oanda_base_url}/v3/instruments/{self._oanda_instrument}/candles"
        headers = {
            "Authorization": f"Bearer {self._oanda_api_key}",
            "Content-Type": "application/json",
        }
        params = {"granularity": "M1", "count": 1, "price": "MBA"}  # Mid, Bid, Ask

        response = self.session.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        candles = data.get("candles", [])
        if not candles:
            raise ValueError("OANDA returned no pricing data for XAU/USD")

        latest = candles[-1]
        mid = latest.get("mid", {})
        bid = latest.get("bid", {})
        ask = latest.get("ask", {})

        price = float(mid.get("c", 0))
        return {
            "symbol": "XAU/USD",
            "price": price,
            "bid": float(bid.get("c", price)),
            "ask": float(ask.get("c", price)),
            "volume_24h": float(latest.get("volume", 0)),
            "high_24h": float(mid.get("h", price)),
            "low_24h": float(mid.get("l", price)),
            "change_pct_24h": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fetch_via_ccxt(
        self, timeframe: TimeFrame, limit: int, start_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch gold-proxy OHLCV via CCXT (PAXG/USDT — gold-backed token).
        PAXG tracks physical gold price 1:1, traded 24/7 on Binance.

        Args:
            timeframe: Candlestick interval
            limit: Number of candles to fetch
            start_time: Optional ISO start datetime

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        ccxt_tf = CCXT_TIMEFRAME_MAP.get(timeframe, "1h")
        since = None
        if start_time:
            dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            since = int(dt.timestamp() * 1000)

        logger.info(
            f"Fetching {self._ccxt_gold_symbol} via CCXT "
            f"(Binance direct, tf={ccxt_tf}, limit={limit})"
        )

        # CCXT returns [[timestamp, open, high, low, close, volume], ...]
        ohlcv = self._ccxt_exchange.fetch_ohlcv(
            self._ccxt_gold_symbol, timeframe=ccxt_tf, since=since, limit=limit
        )

        if not ohlcv:
            raise ValueError(f"CCXT returned no {self._ccxt_gold_symbol} data")

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
        Fetch current gold price via CCXT (PAXG/USDT ticker).

        Returns:
            Dict with price, bid, ask, volume, and timestamp
        """
        ticker = self._ccxt_exchange.fetch_ticker(self._ccxt_gold_symbol)
        price = float(ticker.get("last", 0))
        return {
            "symbol": "XAU/USD",
            "price": price,
            "bid": float(ticker.get("bid", price) or price),
            "ask": float(ticker.get("ask", price) or price),
            "volume_24h": float(ticker.get("baseVolume", 0) or 0),
            "high_24h": float(ticker.get("high", price) or price),
            "low_24h": float(ticker.get("low", price) or price),
            "change_pct_24h": float(ticker.get("percentage", 0) or 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


    def _fetch_via_coingecko(
        self, timeframe: TimeFrame, limit: int, start_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch Gold price via CoinGecko API using PAXG (PAX Gold) as proxy.
        PAXG is backed 1:1 by physical gold — tracks gold spot price closely.
        Free API, no authentication required.

        Args:
            timeframe: Candlestick interval
            limit: Number of candles to fetch
            start_time: Optional ISO start datetime

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        # CoinGecko days parameter determines data granularity
        days_map = {
            TimeFrame.M1: 1, TimeFrame.M5: 1, TimeFrame.M15: 1,
            TimeFrame.M30: 2, TimeFrame.H1: 90, TimeFrame.H4: 90,
            TimeFrame.D1: 365, TimeFrame.W1: 365,
        }
        days = days_map.get(timeframe, 90)

        logger.info(f"Fetching Gold (PAXG) via CoinGecko (days={days})")

        response = self.session.get(
            f"{self._coingecko_base_url}/coins/pax-gold/ohlc",
            params={"vs_currency": "usd", "days": days},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("CoinGecko returned no PAXG (Gold) data")

        # CoinGecko OHLC: [[timestamp_ms, open, high, low, close], ...]
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

        if len(df) > limit:
            df = df.tail(limit)

        return df

    def _fetch_coingecko_current_price(self) -> dict:
        """
        Fetch current Gold price from CoinGecko via PAXG.

        Returns:
            Dict with price details
        """
        response = self.session.get(
            f"{self._coingecko_base_url}/simple/price",
            params={
                "ids": "pax-gold",
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json().get("pax-gold", {})
        price = float(data.get("usd", 0))
        return {
            "symbol": "XAU/USD",
            "price": price,
            "bid": price * 0.9998,
            "ask": price * 1.0002,
            "volume_24h": float(data.get("usd_24h_vol", 0)),
            "high_24h": price * 1.01,
            "low_24h": price * 0.99,
            "change_pct_24h": float(data.get("usd_24h_change", 0)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fetch_via_alpha_vantage(
        self, timeframe: TimeFrame, limit: int
    ) -> pd.DataFrame:
        """
        Fetch Gold (XAU/USD) data from Alpha Vantage FX endpoint.
        Uses physical currency exchange rate: XAU (gold) to USD.
        Free tier: 5 calls/min. Get API key at: https://www.alphavantage.co/support/#api-key

        Args:
            timeframe: Candlestick interval
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        # Alpha Vantage uses FX_DAILY or FX_INTRADAY for commodity pairs
        if timeframe in (TimeFrame.D1, TimeFrame.W1):
            function = "FX_DAILY"
            ts_key = "Time Series FX (Daily)"
        else:
            function = "FX_INTRADAY"
            av_interval_map = {
                TimeFrame.M1: "1min", TimeFrame.M5: "5min",
                TimeFrame.M15: "15min", TimeFrame.M30: "30min",
                TimeFrame.H1: "60min", TimeFrame.H4: "60min",
            }
            interval = av_interval_map.get(timeframe, "60min")
            ts_key = f"Time Series FX (Intraday)"

        params = {
            "function": function,
            "from_symbol": "XAU",
            "to_symbol": "USD",
            "apikey": self._alpha_vantage_key,
        }
        if function == "FX_INTRADAY":
            params["interval"] = interval
            params["outputsize"] = "full" if limit > 100 else "compact"

        logger.info(f"Fetching XAU/USD via Alpha Vantage ({function})")

        response = self.session.get(
            self._alpha_vantage_base_url, params=params, timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if "Error Message" in data or "Note" in data:
            raise ValueError(f"Alpha Vantage error: {data.get('Error Message', data.get('Note', ''))}")

        # Find the time series key (Alpha Vantage key names vary)
        time_series = None
        for key in data:
            if "Time Series" in key:
                time_series = data[key]
                break
        if not time_series:
            raise ValueError("Alpha Vantage returned no Gold data")

        records = []
        for dt_str, values in time_series.items():
            records.append({
                "timestamp": datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": 0.0,  # FX pairs don't have volume in Alpha Vantage
            })

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)  # Alpha Vantage returns newest first
        df.index.name = "datetime"

        if len(df) > limit:
            df = df.tail(limit)

        return df

    def _fetch_alpha_vantage_current_price(self) -> dict:
        """
        Fetch current Gold (XAU/USD) price from Alpha Vantage.

        Returns:
            Dict with price details
        """
        response = self.session.get(
            self._alpha_vantage_base_url,
            params={
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "XAU",
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
            "symbol": "XAU/USD",
            "price": price,
            "bid": bid,
            "ask": ask,
            "volume_24h": 0.0,
            "high_24h": price * 1.01,
            "low_24h": price * 0.99,
            "change_pct_24h": 0.0,
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
        Fetch historical gold OHLCV data with 4-layer fallback:
          1. CoinGecko API (free, PAXG gold-backed token proxy)
          2. Alpha Vantage API (free with API key, XAU/USD FX data)
          3. CCXT (PAXG/USDT — gold-backed token, 24/7 trading)
          4. Yahoo Finance (GC=F gold futures — last resort)

        Args:
            timeframe: Candlestick interval
            start_time: ISO format start datetime (used as period1 epoch)
            end_time: ISO format end datetime (used as period2 epoch)
            limit: Not directly supported by Yahoo; controls range selection

        Returns:
            DataFrame with columns [open, high, low, close, volume], DatetimeIndex
        """
        # --- Layer 1: CoinGecko (free, PAXG gold proxy, no API key needed) ---
        try:
            df = self._fetch_via_coingecko(timeframe, limit, start_time)
            logger.info(f"CoinGecko: fetched {len(df)} Gold (PAXG) candles successfully")
            df = self.validate_dataframe(df)
            self._cache = df
            return df
        except Exception as e:
            logger.warning(f"CoinGecko gold failed: {e}. Falling back to Alpha Vantage.")

        # --- Layer 2: Alpha Vantage (free with API key, XAU/USD FX data) ---
        if self._alpha_vantage_key:
            try:
                df = self._fetch_via_alpha_vantage(timeframe, limit)
                logger.info(f"Alpha Vantage: fetched {len(df)} XAU/USD candles successfully")
                df = self.validate_dataframe(df)
                self._cache = df
                return df
            except Exception as e:
                logger.warning(f"Alpha Vantage gold failed: {e}. Falling back to CCXT.")

        # --- Layer 3: CCXT (PAXG/USDT gold-backed token, 24/7) ---
        if self._ccxt_exchange is not None and self._ccxt_gold_symbol:
            try:
                df = self._fetch_via_ccxt(timeframe, limit, start_time)
                logger.info(f"CCXT: fetched {len(df)} gold (PAXG/USDT) candles successfully")
                # Handle 4h resampling if needed
                if timeframe == TimeFrame.H4:
                    df = df.resample("4h").agg({
                        "open": "first", "high": "max",
                        "low": "min", "close": "last", "volume": "sum",
                    }).dropna()
                df = self.validate_dataframe(df)
                self._cache = df
                return df
            except Exception as e:
                logger.warning(f"CCXT gold fetch failed: {e}. Falling back to Yahoo Finance.")

        # --- Layer 4: Yahoo Finance (GC=F gold futures — last resort) ---
        interval = TIMEFRAME_MAP.get(timeframe, "1h")
        params = {"interval": interval}

        # Use explicit period1/period2 if start/end times are provided
        if start_time:
            dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            params["period1"] = int(dt.timestamp())
        if end_time:
            dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            params["period2"] = int(dt.timestamp())

        # If no explicit dates, use the range map for the timeframe
        if "period1" not in params:
            params["range"] = RANGE_MAP.get(timeframe, "730d")

        logger.info(f"Fetching XAU/USD {interval} candles from Yahoo Finance")
        raw = self._make_request(params)

        # Parse the Yahoo Finance response structure
        timestamps = raw.get("timestamp", [])
        indicators = raw.get("indicators", {})
        quote = indicators.get("quote", [{}])[0]

        records = []
        for i, ts in enumerate(timestamps):
            # Skip entries where OHLC data is None (market closed periods)
            if any(
                quote.get(field, [None])[i] is None
                for field in ["open", "high", "low", "close"]
            ):
                continue

            records.append({
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
                "open": float(quote["open"][i]),
                "high": float(quote["high"][i]),
                "low": float(quote["low"][i]),
                "close": float(quote["close"][i]),
                "volume": float(quote.get("volume", [0])[i] or 0),
            })

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No gold data returned from Yahoo Finance")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df.set_index("timestamp", inplace=True)
        df.index.name = "datetime"

        # If 4h was requested, resample from 1h data
        if timeframe == TimeFrame.H4:
            df = df.resample("4h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        # Validate and clean the data
        df = self.validate_dataframe(df)
        self._cache = df

        logger.info(f"Fetched {len(df)} XAU/USD candles")
        return df

    def fetch_current_price(self) -> dict:
        """
        Fetch current gold price with 4-layer fallback:
          1. CoinGecko API (free, PAXG gold proxy)
          2. Alpha Vantage API (free with API key)
          3. CCXT (PAXG/USDT direct exchange)
          4. Yahoo Finance (GC=F gold futures)

        Returns:
            Dict with price details and metadata
        """
        # --- Layer 1: CoinGecko (free, PAXG gold proxy) ---
        try:
            return self._fetch_coingecko_current_price()
        except Exception as e:
            logger.warning(f"CoinGecko gold price failed: {e}. Trying Alpha Vantage.")

        # --- Layer 2: Alpha Vantage (free with API key) ---
        if self._alpha_vantage_key:
            try:
                return self._fetch_alpha_vantage_current_price()
            except Exception as e:
                logger.warning(f"Alpha Vantage gold price failed: {e}. Trying CCXT.")

        # --- Layer 3: CCXT (PAXG/USDT, direct exchange connection) ---
        if self._ccxt_exchange is not None and self._ccxt_gold_symbol:
            try:
                return self._fetch_ccxt_current_price()
            except Exception as e:
                logger.warning(f"CCXT gold price failed: {e}. Trying Yahoo Finance.")

        # --- Layer 4: Yahoo Finance (GC=F gold futures — last resort) ---
        params = {"interval": "1m", "range": "1d"}
        raw = self._make_request(params)

        # Extract the latest price from meta or last candle
        meta = raw.get("meta", {})
        current_price = meta.get("regularMarketPrice", 0.0)
        prev_close = meta.get("chartPreviousClose", current_price)

        # Calculate 24h change percentage
        change_pct = 0.0
        if prev_close and prev_close != 0:
            change_pct = ((current_price - prev_close) / prev_close) * 100

        return {
            "symbol": "XAU/USD",
            "price": float(current_price),
            "bid": float(current_price),   # Yahoo doesn't provide bid/ask
            "ask": float(current_price),
            "volume_24h": 0.0,             # Volume not reliably available for spot gold
            "high_24h": float(meta.get("regularMarketDayHigh", current_price)),
            "low_24h": float(meta.get("regularMarketDayLow", current_price)),
            "change_pct_24h": round(change_pct, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def fetch_orderbook(self, depth: int = 10) -> dict:
        """
        Gold spot doesn't have a traditional public order book.
        Returns a synthetic order book based on the current price.

        Args:
            depth: Number of synthetic price levels

        Returns:
            Synthetic order book dict
        """
        price_data = self.fetch_current_price()
        price = price_data["price"]

        # Generate synthetic bid/ask levels around current price
        # Spread is approximately 0.3-0.5 USD for gold futures
        spread = 0.30
        bids = [[round(price - spread / 2 - i * 0.10, 2), round(10 + i * 5, 2)]
                 for i in range(depth)]
        asks = [[round(price + spread / 2 + i * 0.10, 2), round(10 + i * 5, 2)]
                 for i in range(depth)]

        return {
            "symbol": "XAU/USD",
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Synthetic order book - gold spot has no public L2 data",
        }
