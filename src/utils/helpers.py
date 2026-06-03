"""
Utility helper functions used across the trading system.
Provides common operations for data manipulation, time handling,
and mathematical computations.
"""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def timestamp_ms() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to millisecond timestamp."""
    return int(dt.timestamp() * 1000)


def round_to_precision(value: float, precision: int) -> float:
    """Round a float to the specified decimal precision."""
    return round(value, precision)


def round_to_step(value: float, step: float) -> float:
    """Round a value down to the nearest step increment."""
    return round(value - (value % step), 10)


def calculate_pct_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / abs(old_value)) * 100.0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def generate_order_id(symbol: str, side: str, timestamp: Optional[int] = None) -> str:
    """Generate a unique order ID based on symbol, side, and timestamp."""
    ts = timestamp or timestamp_ms()
    raw = f"{symbol}_{side}_{ts}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame by handling missing data.
    Forward-fills then backward-fills NaN values.
    """
    # Forward fill missing values first
    df = df.ffill()
    # Backward fill any remaining NaNs at the start
    df = df.bfill()
    return df


def align_timeframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = "inner"
) -> tuple:
    """
    Align two DataFrames to the same time index.
    Useful for aligning BTC and Gold data for correlation analysis.

    Args:
        df1: First DataFrame with datetime index
        df2: Second DataFrame with datetime index
        method: Join method ('inner', 'outer', 'left', 'right')

    Returns:
        Tuple of (aligned_df1, aligned_df2)
    """
    # Merge on index to align timestamps
    combined = df1.join(df2, how=method, lsuffix="_1", rsuffix="_2")
    # Split back and forward-fill any gaps from outer join
    cols1 = [c for c in combined.columns if c.endswith("_1")]
    cols2 = [c for c in combined.columns if c.endswith("_2")]

    aligned1 = combined[cols1].rename(columns={c: c[:-2] for c in cols1})
    aligned2 = combined[cols2].rename(columns={c: c[:-2] for c in cols2})

    return normalize_dataframe(aligned1), normalize_dataframe(aligned2)


def calculate_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 30
) -> pd.Series:
    """
    Calculate rolling correlation between two price series.
    Used for BTC-Gold correlation analysis.
    """
    return series1.rolling(window=window).corr(series2)


def detect_outliers(series: pd.Series, n_std: float = 3.0) -> pd.Series:
    """
    Detect outliers in a series using z-score method.
    Returns boolean Series where True indicates an outlier.
    """
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    return z_scores.abs() > n_std


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] and datetime index
        timeframe: Target timeframe string (e.g., '5min', '1h', '1D')

    Returns:
        Resampled OHLCV DataFrame
    """
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return resampled.dropna()


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as USD currency string."""
    return f"${value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a number as percentage string."""
    return f"{value:.{decimals}f}%"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max bounds."""
    return max(min_val, min(value, max_val))


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period).mean()


def annualized_return(total_return_pct: float, days: int) -> float:
    """Calculate annualized return from total return and holding period."""
    if days <= 0:
        return 0.0
    return ((1 + total_return_pct / 100) ** (365.0 / days) - 1) * 100


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio from a series of returns.
    Assumes daily returns; annualizes by sqrt(252).
    """
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio (penalizes only downside volatility).
    """
    excess = returns - risk_free_rate / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / downside.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown percentage from an equity curve.
    Returns a negative percentage.
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return float(drawdown.min()) * 100
