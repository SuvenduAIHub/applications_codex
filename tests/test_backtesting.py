"""
Unit tests for the backtesting engine and performance analysis.
Tests backtest execution, metric calculation, and result validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.performance import PerformanceAnalyzer
from src.strategies.trend_following import TrendFollowingStrategy


def _make_equity_curve(n: int = 252, initial: float = 100000.0) -> pd.Series:
    """Generate a synthetic equity curve for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = np.random.normal(0.0005, 0.015, n)
    equity = initial * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


def _make_trade_log(n_trades: int = 50) -> list:
    """Generate a synthetic trade log for testing."""
    np.random.seed(42)
    trades = []
    for i in range(n_trades):
        pnl = np.random.normal(50, 200)
        trades.append({
            "symbol": "BTC/USDT" if i % 2 == 0 else "XAU/USD",
            "side": "buy",
            "quantity": 0.01,
            "entry_price": 50000,
            "exit_price": 50000 + pnl * 100,
            "pnl": pnl,
            "pnl_pct": pnl / 500,
            "commission": 5.0,
            "reason": "signal",
            "closed_at": "2024-06-15T12:00:00",
            "duration_seconds": 3600 * np.random.uniform(1, 48),
        })
    return trades


class TestPerformanceAnalyzer:
    """Tests for the performance analysis module."""

    def test_all_metrics_returned(self):
        """calculate_all_metrics should return all metric categories."""
        equity = _make_equity_curve()
        trades = _make_trade_log()
        analyzer = PerformanceAnalyzer(equity, trades)
        metrics = analyzer.calculate_all_metrics()

        assert "return_metrics" in metrics
        assert "risk_metrics" in metrics
        assert "trade_metrics" in metrics
        assert "drawdown_metrics" in metrics
        assert "ratio_metrics" in metrics

    def test_total_return_calculation(self):
        """Total return should be correctly calculated."""
        equity = _make_equity_curve()
        analyzer = PerformanceAnalyzer(equity, [])
        ret = analyzer._return_metrics()
        # Total return should be (final - initial) / initial * 100
        expected = ((equity.iloc[-1] - 100000) / 100000) * 100
        assert ret["total_return_pct"] == pytest.approx(expected, rel=0.01)

    def test_sharpe_ratio_sign(self):
        """Sharpe ratio should be positive for a positive-return strategy."""
        # Create a consistently positive equity curve
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        equity = pd.Series(np.linspace(100000, 120000, 100), index=dates)
        analyzer = PerformanceAnalyzer(equity, [])
        ratios = analyzer._ratio_metrics()
        assert ratios["sharpe_ratio"] > 0

    def test_max_drawdown_is_negative(self):
        """Max drawdown should be zero or negative (it measures decline)."""
        equity = _make_equity_curve()
        analyzer = PerformanceAnalyzer(equity, [])
        dd = analyzer._drawdown_metrics()
        assert dd["max_drawdown_pct"] <= 0

    def test_win_rate_calculation(self):
        """Win rate should be correctly calculated from trade log."""
        trades = _make_trade_log(100)
        equity = _make_equity_curve()
        analyzer = PerformanceAnalyzer(equity, trades)
        trade_metrics = analyzer._trade_metrics()

        wins = len([t for t in trades if t["pnl"] > 0])
        expected_wr = (wins / len(trades)) * 100
        assert trade_metrics["win_rate_pct"] == pytest.approx(expected_wr, abs=0.1)

    def test_profit_factor_positive(self):
        """Profit factor should be > 0 when there are both wins and losses."""
        trades = _make_trade_log()
        equity = _make_equity_curve()
        analyzer = PerformanceAnalyzer(equity, trades)
        tm = analyzer._trade_metrics()
        assert tm["profit_factor"] > 0

    def test_empty_trade_log(self):
        """Should handle empty trade log gracefully."""
        equity = _make_equity_curve()
        analyzer = PerformanceAnalyzer(equity, [])
        tm = analyzer._trade_metrics()
        assert tm["total_trades"] == 0
        assert tm["win_rate_pct"] == 0

    def test_report_generation(self):
        """Performance report should generate a non-empty string."""
        equity = _make_equity_curve()
        trades = _make_trade_log()
        analyzer = PerformanceAnalyzer(equity, trades)
        report = analyzer.generate_report()
        assert len(report) > 100
        assert "BACKTESTING PERFORMANCE REPORT" in report
        assert "Sharpe Ratio" in report
        assert "Max Drawdown" in report

    def test_sortino_ratio(self):
        """Sortino ratio should be calculable for standard equity curves."""
        equity = _make_equity_curve()
        analyzer = PerformanceAnalyzer(equity, [])
        ratios = analyzer._ratio_metrics()
        # Sortino should be a finite number
        assert np.isfinite(ratios["sortino_ratio"])

    def test_max_consecutive_wins(self):
        """Should correctly count maximum consecutive wins."""
        pnls = [10, 20, 30, -5, 10, 20, 30, 40, 50, -10]
        result = PerformanceAnalyzer._max_consecutive(pnls, positive=True)
        assert result == 5  # The streak of 10,20,30,40,50

    def test_max_consecutive_losses(self):
        """Should correctly count maximum consecutive losses."""
        pnls = [10, -5, -10, -15, 20, -5, -10]
        result = PerformanceAnalyzer._max_consecutive(pnls, positive=False)
        assert result == 3  # The streak of -5,-10,-15
