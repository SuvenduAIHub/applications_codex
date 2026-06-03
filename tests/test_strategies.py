"""
Unit tests for the trading strategy module.
Tests signal generation for trend-following, mean-reversion, and breakout strategies.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import StrategyConfig
from src.indicators.technical import add_all_indicators
from src.indicators.regime import compute_regime_features
from src.strategies.base_strategy import Signal, TradeSignal
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy


def _make_enriched_ohlcv(n: int = 200, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data with all indicators for strategy testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    returns = np.random.normal(0.0002, 0.01, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.uniform(100, 10000, n)

    df = pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    enriched = add_all_indicators(df)
    enriched = compute_regime_features(enriched)
    return enriched


class TestTrendFollowingStrategy:
    """Tests for the trend-following strategy."""

    def test_returns_trade_signal(self):
        """Strategy should always return a TradeSignal object."""
        strategy = TrendFollowingStrategy()
        df = _make_enriched_ohlcv()
        signal = strategy.generate_signal(df, "BTC/USDT")
        assert isinstance(signal, TradeSignal)

    def test_signal_has_valid_type(self):
        """Signal should be one of the defined Signal enum values."""
        strategy = TrendFollowingStrategy()
        df = _make_enriched_ohlcv()
        signal = strategy.generate_signal(df, "BTC/USDT")
        assert signal.signal in Signal

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        strategy = TrendFollowingStrategy()
        df = _make_enriched_ohlcv()
        signal = strategy.generate_signal(df, "BTC/USDT")
        assert 0.0 <= signal.confidence <= 1.0

    def test_hold_signal_for_insufficient_data(self):
        """Should return HOLD for very short data."""
        strategy = TrendFollowingStrategy()
        df = _make_enriched_ohlcv(n=2)
        signal = strategy.generate_signal(df, "BTC/USDT")
        assert signal.signal == Signal.HOLD

    def test_strategy_name(self):
        """Strategy name should be 'trend_following'."""
        strategy = TrendFollowingStrategy()
        assert strategy.get_name() == "trend_following"

    def test_cooldown_mechanism(self):
        """Strategy with non-zero cooldown should respect trade cooldown."""
        # Default cooldown is 0 (scalping mode), so use explicit cooldown for test
        config = StrategyConfig(trade_cooldown_candles=1)
        strategy = TrendFollowingStrategy(config)
        strategy.on_trade_executed()
        assert not strategy.should_trade()  # Should be in cooldown

    def test_cooldown_expires(self):
        """Cooldown should expire after configured number of candles."""
        config = StrategyConfig(trade_cooldown_candles=2)
        strategy = TrendFollowingStrategy(config)
        strategy.on_trade_executed()
        assert not strategy.should_trade()
        assert not strategy.should_trade()
        assert strategy.should_trade()  # Cooldown expired


class TestMeanReversionStrategy:
    """Tests for the mean-reversion strategy."""

    def test_returns_trade_signal(self):
        """Strategy should always return a TradeSignal object."""
        strategy = MeanReversionStrategy()
        df = _make_enriched_ohlcv()
        signal = strategy.generate_signal(df, "XAU/USD")
        assert isinstance(signal, TradeSignal)

    def test_strategy_name(self):
        """Strategy name should be 'mean_reversion'."""
        strategy = MeanReversionStrategy()
        assert strategy.get_name() == "mean_reversion"


class TestBreakoutStrategy:
    """Tests for the breakout strategy."""

    def test_returns_trade_signal(self):
        """Strategy should always return a TradeSignal object."""
        strategy = BreakoutStrategy()
        df = _make_enriched_ohlcv()
        signal = strategy.generate_signal(df, "BTC/USDT")
        assert isinstance(signal, TradeSignal)

    def test_strategy_name(self):
        """Strategy name should be 'breakout'."""
        strategy = BreakoutStrategy()
        assert strategy.get_name() == "breakout"

    def test_hold_for_short_data(self):
        """Should return HOLD when data is shorter than lookback."""
        strategy = BreakoutStrategy(lookback=50)
        df = _make_enriched_ohlcv(n=30)
        signal = strategy.generate_signal(df, "BTC/USDT")
        assert signal.signal == Signal.HOLD


class TestEnsembleStrategy:
    """Tests for the ensemble strategy."""

    def test_ensemble_returns_signal(self):
        """Ensemble should return a valid TradeSignal."""
        strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
        ]
        ensemble = EnsembleStrategy(strategies)
        df = _make_enriched_ohlcv()
        signal = ensemble.generate_signal(df, "BTC/USDT")
        assert isinstance(signal, TradeSignal)

    def test_ensemble_name(self):
        """Ensemble strategy name should be 'ensemble'."""
        ensemble = EnsembleStrategy([TrendFollowingStrategy()])
        assert ensemble.get_name() == "ensemble"

    def test_ensemble_metadata_contains_sub_signals(self):
        """Ensemble signal metadata should contain sub-strategy details."""
        strategies = [TrendFollowingStrategy(), MeanReversionStrategy()]
        ensemble = EnsembleStrategy(strategies, min_consensus=0.0)
        df = _make_enriched_ohlcv()
        signal = ensemble.generate_signal(df, "BTC/USDT")
        # Metadata should contain information about sub-strategy signals
        assert "sub_signals" in signal.metadata or signal.signal == Signal.HOLD

    def test_ensemble_performance_tracking(self):
        """Ensemble should track strategy performance for adaptive weighting."""
        strategies = [TrendFollowingStrategy(), MeanReversionStrategy()]
        ensemble = EnsembleStrategy(strategies)
        # Record some performance
        ensemble.update_performance("trend_following", 100.0)
        ensemble.update_performance("trend_following", 50.0)
        ensemble.update_performance("mean_reversion", -30.0)
        # Weights should have been updated
        assert len(ensemble.strategy_performance["trend_following"]) == 2
        assert len(ensemble.strategy_performance["mean_reversion"]) == 1
