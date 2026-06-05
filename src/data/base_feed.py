"""
Abstract base class for all market data feeds.
Defines the interface that BTC and Gold feeds must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from config.settings import DataFeedConfig, TimeFrame


class BaseDataFeed(ABC):
    """
    Abstract base class for market data feeds.
    All concrete feed implementations (BTC, Gold) must inherit from this
    and implement the required methods for data retrieval.
    """

    def __init__(self, symbol: str, config: DataFeedConfig):
        """
        Initialize the data feed.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT", "XAU/USD")
            config: Data feed configuration settings
        """
        self.symbol = symbol
        self.config = config
        self._cache: Optional[pd.DataFrame] = None

    @abstractmethod
    def fetch_historical(
        self,
        timeframe: TimeFrame,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candlestick data.

        Args:
            timeframe: Candlestick timeframe (1m, 5m, 1h, 1d, etc.)
            start_time: Start datetime string (ISO format)
            end_time: End datetime string (ISO format)
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex (UTC)
        """
        pass

    @abstractmethod
    def fetch_current_price(self) -> dict:
        """
        Fetch the current/latest price data.

        Returns:
            Dictionary with keys: {price, bid, ask, volume_24h, timestamp}
        """
        pass

    @abstractmethod
    def fetch_orderbook(self, depth: int = 10) -> dict:
        """
        Fetch the current order book.

        Args:
            depth: Number of price levels to fetch on each side

        Returns:
            Dictionary with keys: {bids: [[price, qty], ...], asks: [[price, qty], ...]}
        """
        pass

    def get_cached_data(self) -> Optional[pd.DataFrame]:
        """Return cached data if available."""
        return self._cache

    def clear_cache(self) -> None:
        """Clear the local data cache."""
        self._cache = None

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean a raw OHLCV DataFrame.
        Ensures required columns exist, handles missing data,
        and sorts by datetime index.

        Args:
            df: Raw DataFrame to validate

        Returns:
            Cleaned and validated DataFrame
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure numeric types for price/volume columns
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by index (datetime) ascending
        df = df.sort_index()

        # Forward-fill then backward-fill missing values
        df = df.ffill().bfill()

        # Remove any rows where all OHLCV values are NaN
        df = df.dropna(subset=required_cols, how="all")

        # Validate OHLCV consistency: high >= low, high >= open/close, etc.
        df = df[df["high"] >= df["low"]]

        return df
