"""
Performance analysis and metrics calculation for backtesting results.
Computes CAGR, Sharpe ratio, Sortino ratio, max drawdown, win rate,
profit factor, and other key trading metrics.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class PerformanceAnalyzer:
    """
    Analyzes backtesting results and computes comprehensive performance metrics.

    Metrics calculated:
        - Total return and CAGR
        - Sharpe ratio (risk-adjusted return)
        - Sortino ratio (downside risk-adjusted return)
        - Maximum drawdown and duration
        - Win rate and loss rate
        - Profit factor (gross profit / gross loss)
        - Average win/loss and their ratio
        - Calmar ratio (CAGR / max drawdown)
        - Trade statistics (count, duration, frequency)
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trade_log: List[dict],
        initial_balance: float = 100000.0,
        risk_free_rate: float = 0.04,  # Annual risk-free rate (e.g., T-bills)
    ):
        """
        Initialize the performance analyzer.

        Args:
            equity_curve: Series of portfolio equity values (DatetimeIndex)
            trade_log: List of trade result dictionaries from backtesting
            initial_balance: Starting portfolio balance
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation
        """
        self.equity_curve = equity_curve
        self.trade_log = trade_log
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(self) -> dict:
        """
        Calculate all performance metrics.

        Returns:
            Comprehensive metrics dictionary
        """
        return {
            "return_metrics": self._return_metrics(),
            "risk_metrics": self._risk_metrics(),
            "trade_metrics": self._trade_metrics(),
            "drawdown_metrics": self._drawdown_metrics(),
            "ratio_metrics": self._ratio_metrics(),
        }

    def _return_metrics(self) -> dict:
        """Calculate return-based metrics."""
        if len(self.equity_curve) < 2:
            return {"total_return_pct": 0, "cagr_pct": 0, "total_pnl": 0}

        final_value = self.equity_curve.iloc[-1]
        total_return_pct = ((final_value - self.initial_balance) / self.initial_balance) * 100

        # Calculate CAGR (Compound Annual Growth Rate)
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25 if days > 0 else 1

        if years > 0 and final_value > 0:
            cagr = ((final_value / self.initial_balance) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "total_pnl": round(final_value - self.initial_balance, 2),
            "final_equity": round(final_value, 2),
            "initial_balance": self.initial_balance,
            "trading_days": days,
        }

    def _risk_metrics(self) -> dict:
        """Calculate risk-based metrics (volatility, etc.)."""
        if len(self.equity_curve) < 2:
            return {"annual_volatility_pct": 0, "daily_volatility_pct": 0}

        # Daily returns from equity curve
        returns = self.equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return {"annual_volatility_pct": 0, "daily_volatility_pct": 0}

        daily_vol = float(returns.std())
        annual_vol = daily_vol * np.sqrt(252)

        # Value at Risk (95th percentile)
        var_95 = float(np.percentile(returns, 5)) * 100  # 5th percentile of returns

        # Conditional VaR (Expected Shortfall) - average of returns below VaR
        cvar_returns = returns[returns <= np.percentile(returns, 5)]
        cvar_95 = float(cvar_returns.mean()) * 100 if len(cvar_returns) > 0 else 0

        return {
            "annual_volatility_pct": round(annual_vol * 100, 2),
            "daily_volatility_pct": round(daily_vol * 100, 4),
            "var_95_pct": round(var_95, 2),
            "cvar_95_pct": round(cvar_95, 2),
            "skewness": round(float(returns.skew()), 4),
            "kurtosis": round(float(returns.kurtosis()), 4),
        }

    def _trade_metrics(self) -> dict:
        """Calculate trade-based metrics (win rate, profit factor, etc.)."""
        # Default empty-trade metrics dict with all keys the report template uses
        _empty = {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "breakeven_trades": 0, "win_rate_pct": 0, "profit_factor": 0,
            "gross_profit": 0, "gross_loss": 0, "net_profit": 0,
            "avg_trade_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "win_loss_ratio": 0, "largest_win": 0, "largest_loss": 0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "avg_duration_hours": 0,
        }
        if not self.trade_log:
            return _empty

        pnls = [t["pnl"] for t in self.trade_log if "pnl" in t]
        if not pnls:
            return _empty

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakeven = [p for p in pnls if p == 0]

        total_trades = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # Max consecutive wins/losses
        max_consec_wins = self._max_consecutive(pnls, positive=True)
        max_consec_losses = self._max_consecutive(pnls, positive=False)

        # Average trade duration
        durations = [t.get("duration_seconds", 0) for t in self.trade_log]
        avg_duration_hours = np.mean(durations) / 3600 if durations else 0

        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "breakeven_trades": len(breakeven),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(gross_profit - gross_loss, 2),
            "avg_trade_pnl": round(np.mean(pnls), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "largest_win": round(max(wins), 2) if wins else 0,
            "largest_loss": round(min(losses), 2) if losses else 0,
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "avg_trade_duration_hours": round(avg_duration_hours, 2),
        }

    def _drawdown_metrics(self) -> dict:
        """Calculate drawdown metrics (max drawdown, duration, etc.)."""
        if len(self.equity_curve) < 2:
            return {"max_drawdown_pct": 0, "max_drawdown_duration_days": 0}

        # Running maximum of equity
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        max_dd = float(drawdown.min()) * 100

        # Find drawdown periods
        is_dd = drawdown < 0
        dd_starts = is_dd & ~is_dd.shift(1, fill_value=False)
        dd_ends = ~is_dd & is_dd.shift(1, fill_value=False)

        # Longest drawdown duration
        max_dd_duration = 0
        current_dd_start = None
        for i, (idx, in_dd) in enumerate(is_dd.items()):
            if in_dd and current_dd_start is None:
                current_dd_start = idx
            elif not in_dd and current_dd_start is not None:
                duration = (idx - current_dd_start).days
                max_dd_duration = max(max_dd_duration, duration)
                current_dd_start = None

        # If still in drawdown at the end
        if current_dd_start is not None:
            duration = (self.equity_curve.index[-1] - current_dd_start).days
            max_dd_duration = max(max_dd_duration, duration)

        # Average drawdown
        avg_dd = float(drawdown[drawdown < 0].mean()) * 100 if (drawdown < 0).any() else 0

        return {
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_duration_days": max_dd_duration,
            "avg_drawdown_pct": round(avg_dd, 2),
            "current_drawdown_pct": round(float(drawdown.iloc[-1]) * 100, 2),
        }

    def _ratio_metrics(self) -> dict:
        """Calculate risk-adjusted performance ratios."""
        if len(self.equity_curve) < 2:
            return {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0}

        returns = self.equity_curve.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0}

        # Sharpe Ratio: (mean return - risk_free) / std(returns) * sqrt(252)
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())

        # Sortino Ratio: uses only downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = float(downside_returns.std()) if len(downside_returns) > 0 else 0
        sortino = float(np.sqrt(252) * excess_returns.mean() / downside_std) if downside_std > 0 else 0

        # Calmar Ratio: CAGR / |max drawdown|
        return_metrics = self._return_metrics()
        dd_metrics = self._drawdown_metrics()
        cagr = return_metrics["cagr_pct"]
        max_dd = abs(dd_metrics["max_drawdown_pct"])
        calmar = cagr / max_dd if max_dd > 0 else 0

        # Information Ratio (simplified - assumes benchmark is 0)
        info_ratio = float(np.sqrt(252) * returns.mean() / returns.std())

        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "information_ratio": round(info_ratio, 4),
        }

    @staticmethod
    def _max_consecutive(pnls: List[float], positive: bool) -> int:
        """
        Calculate maximum consecutive wins or losses.

        Args:
            pnls: List of trade PnLs
            positive: True for consecutive wins, False for consecutive losses

        Returns:
            Maximum consecutive count
        """
        max_count = 0
        current_count = 0
        for pnl in pnls:
            if (positive and pnl > 0) or (not positive and pnl < 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    def generate_report(self) -> str:
        """
        Generate a human-readable performance report.

        Returns:
            Formatted string report
        """
        metrics = self.calculate_all_metrics()
        ret = metrics["return_metrics"]
        risk = metrics["risk_metrics"]
        trade = metrics["trade_metrics"]
        dd = metrics["drawdown_metrics"]
        ratios = metrics["ratio_metrics"]

        report = f"""
{'='*60}
           BACKTESTING PERFORMANCE REPORT
{'='*60}

RETURN METRICS
  Total Return:          {ret['total_return_pct']:>10.2f}%
  CAGR:                  {ret['cagr_pct']:>10.2f}%
  Total P&L:            ${ret['total_pnl']:>12,.2f}
  Final Equity:         ${ret['final_equity']:>12,.2f}
  Trading Days:          {ret['trading_days']:>10d}

RISK METRICS
  Annual Volatility:     {risk['annual_volatility_pct']:>10.2f}%
  VaR (95%):             {risk['var_95_pct']:>10.2f}%
  CVaR (95%):            {risk['cvar_95_pct']:>10.2f}%
  Skewness:              {risk['skewness']:>10.4f}
  Kurtosis:              {risk['kurtosis']:>10.4f}

RISK-ADJUSTED RATIOS
  Sharpe Ratio:          {ratios['sharpe_ratio']:>10.4f}
  Sortino Ratio:         {ratios['sortino_ratio']:>10.4f}
  Calmar Ratio:          {ratios['calmar_ratio']:>10.4f}

DRAWDOWN METRICS
  Max Drawdown:          {dd['max_drawdown_pct']:>10.2f}%
  Max DD Duration:       {dd['max_drawdown_duration_days']:>10d} days
  Avg Drawdown:          {dd['avg_drawdown_pct']:>10.2f}%

TRADE METRICS
  Total Trades:          {trade['total_trades']:>10d}
  Win Rate:              {trade['win_rate_pct']:>10.2f}%
  Profit Factor:         {trade['profit_factor']:>10.2f}
  Avg Trade P&L:        ${trade['avg_trade_pnl']:>12,.2f}
  Avg Win:              ${trade['avg_win']:>12,.2f}
  Avg Loss:             ${trade['avg_loss']:>12,.2f}
  Win/Loss Ratio:        {trade['win_loss_ratio']:>10.2f}
  Largest Win:          ${trade.get('largest_win', 0):>12,.2f}
  Largest Loss:         ${trade.get('largest_loss', 0):>12,.2f}
  Max Consec. Wins:      {trade['max_consecutive_wins']:>10d}
  Max Consec. Losses:    {trade['max_consecutive_losses']:>10d}

{'='*60}
"""
        return report
