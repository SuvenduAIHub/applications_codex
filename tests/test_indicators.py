"""
Unit tests for the technical indicators module.
Tests RSI, MACD, Bollinger Bands, ATR, and feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.technical import (
    add_all_indicators,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
    compute_volume_features,
)


def _make_ohlcv(n: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    # Random walk for close prices
    returns = np.random.normal(0.0002, 0.01, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.uniform(100, 10000, n)

    return pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_returns_correct_length(self):
        """SMA output should have same length as input."""
        series = pd.Series(range(100), dtype=float)
        result = compute_sma(series, 20)
        assert len(result) == 100

    def test_sma_first_values_are_nan(self):
        """First (period-1) values should be NaN."""
        series = pd.Series(range(20), dtype=float)
        result = compute_sma(series, 5)
        assert pd.isna(result.iloc[3])  # 4th element (0-indexed) should still be NaN
        assert not pd.isna(result.iloc[4])  # 5th element should have a value

    def test_sma_known_value(self):
        """Test SMA with known values."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = compute_sma(series, 3)
        assert result.iloc[2] == pytest.approx(20.0)  # (10+20+30)/3
        assert result.iloc[4] == pytest.approx(40.0)  # (30+40+50)/3


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_returns_correct_length(self):
        """EMA output should have same length as input."""
        series = pd.Series(range(100), dtype=float)
        result = compute_ema(series, 20)
        assert len(result) == 100

    def test_ema_no_nan(self):
        """EMA should not produce NaN values (uses ewm with adjust=False)."""
        series = pd.Series(range(50), dtype=float)
        result = compute_ema(series, 10)
        assert not result.isna().any()


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_range(self):
        """RSI values should be between 0 and 100."""
        df = _make_ohlcv(200)
        rsi = compute_rsi(df["close"], period=14)
        # After warmup period, values should be in valid range
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_known_trend(self):
        """RSI should be high (>50) during uptrends."""
        # Monotonically increasing prices — longer series for RSI warmup
        uptrend = pd.Series(np.linspace(100, 200, 100))
        rsi = compute_rsi(uptrend, period=14)
        # RSI should be above or equal to 50 during consistent uptrend
        assert rsi.iloc[-1] >= 50


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_output_columns(self):
        """MACD should return DataFrame with correct columns."""
        df = _make_ohlcv(100)
        macd = compute_macd(df["close"])
        assert "macd" in macd.columns
        assert "macd_signal" in macd.columns
        assert "macd_histogram" in macd.columns

    def test_macd_histogram_is_difference(self):
        """MACD histogram = MACD line - signal line."""
        df = _make_ohlcv(100)
        macd = compute_macd(df["close"])
        # Histogram should equal MACD - signal (approximately)
        diff = macd["macd"] - macd["macd_signal"]
        np.testing.assert_array_almost_equal(
            macd["macd_histogram"].dropna().values,
            diff.dropna().values,
            decimal=10,
        )


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bb_output_columns(self):
        """Bollinger Bands should return all expected columns."""
        df = _make_ohlcv(50)
        bb = compute_bollinger_bands(df["close"])
        expected = {"bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct"}
        assert set(bb.columns) == expected

    def test_bb_ordering(self):
        """Upper band should always be above lower band."""
        df = _make_ohlcv(100)
        bb = compute_bollinger_bands(df["close"])
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_bb_middle_is_sma(self):
        """Middle band should equal the SMA."""
        df = _make_ohlcv(50)
        bb = compute_bollinger_bands(df["close"], period=20)
        sma = compute_sma(df["close"], 20)
        valid = bb["bb_middle"].dropna()
        np.testing.assert_array_almost_equal(valid.values, sma.dropna().values)


class TestATR:
    """Tests for Average True Range."""

    def test_atr_positive(self):
        """ATR should always be positive (it's a volatility measure)."""
        df = _make_ohlcv(100)
        atr = compute_atr(df["high"], df["low"], df["close"])
        valid = atr.dropna()
        assert (valid > 0).all()


class TestVolumeFeatures:
    """Tests for volume-based features."""

    def test_volume_features_output(self):
        """Volume features should return expected columns."""
        df = _make_ohlcv(50)
        vol = compute_volume_features(df["volume"], df["close"])
        expected = {"volume_ma", "volume_ratio", "obv", "vwap"}
        assert set(vol.columns) == expected


class TestAddAllIndicators:
    """Tests for the combined indicator function."""

    def test_all_indicators_adds_columns(self):
        """add_all_indicators should add many new columns to the DataFrame."""
        df = _make_ohlcv(200)
        result = add_all_indicators(df)
        # Should have more columns than original OHLCV (5 columns)
        assert len(result.columns) > 20
        # Should preserve original data
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_all_indicators_no_index_change(self):
        """Adding indicators should not change the DataFrame index."""
        df = _make_ohlcv(100)
        result = add_all_indicators(df)
        assert len(result) == len(df)
        assert result.index.equals(df.index)
