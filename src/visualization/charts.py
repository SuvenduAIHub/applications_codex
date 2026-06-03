"""
Visualization module for generating trading charts and performance reports.
Produces equity curves, drawdown charts, correlation heatmaps,
and trade analysis visualizations.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


class TradingVisualizer:
    """
    Generates publication-quality charts for backtesting results
    and live trading analysis.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save generated chart images
        """
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "charts"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set default style for all charts
        plt.style.use("dark_background")
        self.colors = {
            "primary": "#e94560",
            "secondary": "#0f3460",
            "positive": "#00d4aa",
            "negative": "#e94560",
            "neutral": "#888888",
            "btc": "#f7931a",
            "gold": "#ffd700",
            "bg": "#1a1a2e",
        }

    def plot_equity_curve(
        self,
        equity_data: List[dict],
        title: str = "Portfolio Equity Curve",
        filename: str = "equity_curve.png",
    ) -> str:
        """
        Generate an equity curve chart showing portfolio value over time.

        Args:
            equity_data: List of dicts with 'timestamp' and 'equity' keys
            title: Chart title
            filename: Output filename

        Returns:
            Path to the saved chart image
        """
        if not equity_data:
            logger.warning("No equity data to plot")
            return ""

        df = pd.DataFrame(equity_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor=self.colors["bg"])
        ax.set_facecolor(self.colors["bg"])

        # Plot equity line
        ax.plot(df.index, df["equity"], color=self.colors["positive"], linewidth=1.5, label="Portfolio Value")

        # Add initial balance reference line
        initial = df["equity"].iloc[0]
        ax.axhline(y=initial, color=self.colors["neutral"], linestyle="--", alpha=0.5, label="Initial Balance")

        # Fill area between equity and initial balance
        ax.fill_between(
            df.index, df["equity"], initial,
            where=(df["equity"] >= initial), alpha=0.15, color=self.colors["positive"]
        )
        ax.fill_between(
            df.index, df["equity"], initial,
            where=(df["equity"] < initial), alpha=0.15, color=self.colors["negative"]
        )

        ax.set_title(title, fontsize=16, color="white", pad=15)
        ax.set_xlabel("Date", fontsize=12, color="#888")
        ax.set_ylabel("Portfolio Value ($)", fontsize=12, color="#888")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.2)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)

        logger.info(f"Equity curve saved to {filepath}")
        return filepath

    def plot_drawdown(
        self,
        equity_data: List[dict],
        title: str = "Drawdown Chart",
        filename: str = "drawdown_chart.png",
    ) -> str:
        """
        Generate a drawdown chart showing portfolio drawdown percentage over time.

        Args:
            equity_data: List of dicts with 'timestamp' and 'equity' keys
            title: Chart title
            filename: Output filename

        Returns:
            Path to the saved chart image
        """
        if not equity_data:
            return ""

        df = pd.DataFrame(equity_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Calculate drawdown
        cummax = df["equity"].cummax()
        drawdown = ((df["equity"] - cummax) / cummax) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor=self.colors["bg"],
                                         gridspec_kw={"height_ratios": [2, 1]})

        # Top: Equity curve
        ax1.set_facecolor(self.colors["bg"])
        ax1.plot(df.index, df["equity"], color=self.colors["positive"], linewidth=1.5)
        ax1.plot(df.index, cummax, color=self.colors["neutral"], linewidth=1, linestyle="--", alpha=0.5)
        ax1.set_title(title, fontsize=16, color="white", pad=15)
        ax1.set_ylabel("Portfolio Value ($)", color="#888")
        ax1.grid(True, alpha=0.2)

        # Bottom: Drawdown area
        ax2.set_facecolor(self.colors["bg"])
        ax2.fill_between(df.index, drawdown, 0, color=self.colors["negative"], alpha=0.4)
        ax2.plot(df.index, drawdown, color=self.colors["negative"], linewidth=1)
        ax2.set_ylabel("Drawdown (%)", color="#888")
        ax2.set_xlabel("Date", color="#888")
        ax2.grid(True, alpha=0.2)

        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax2.annotate(
            f"Max DD: {max_dd_val:.2f}%",
            xy=(max_dd_idx, max_dd_val),
            xytext=(max_dd_idx, max_dd_val - 2),
            fontsize=10, color="white",
            arrowprops=dict(arrowstyle="->", color="white"),
        )

        fig.autofmt_xdate()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)

        logger.info(f"Drawdown chart saved to {filepath}")
        return filepath

    def plot_correlation_heatmap(
        self,
        btc_data: pd.DataFrame,
        gold_data: pd.DataFrame,
        title: str = "BTC-Gold Correlation Heatmap",
        filename: str = "correlation_heatmap.png",
    ) -> str:
        """
        Generate a correlation heatmap between BTC and Gold features.

        Args:
            btc_data: BTC OHLCV DataFrame with indicators
            gold_data: Gold OHLCV DataFrame with indicators
            title: Chart title
            filename: Output filename

        Returns:
            Path to the saved chart image
        """
        # Select key features for correlation analysis
        feature_cols = ["close", "volume", "rsi", "macd", "atr", "bb_width",
                        "return_1", "hist_volatility"]

        btc_features = btc_data[[c for c in feature_cols if c in btc_data.columns]].copy()
        gold_features = gold_data[[c for c in feature_cols if c in gold_data.columns]].copy()

        # Rename columns to distinguish BTC vs Gold
        btc_features.columns = [f"BTC_{c}" for c in btc_features.columns]
        gold_features.columns = [f"Gold_{c}" for c in gold_features.columns]

        # Combine and calculate correlation matrix
        combined = pd.concat([btc_features, gold_features], axis=1).dropna()

        if combined.empty:
            logger.warning("No overlapping data for correlation heatmap")
            return ""

        corr_matrix = combined.corr()

        fig, ax = plt.subplots(figsize=(14, 10), facecolor=self.colors["bg"])
        ax.set_facecolor(self.colors["bg"])

        # Generate heatmap using seaborn
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 8},
        )

        ax.set_title(title, fontsize=16, color="white", pad=15)

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)

        logger.info(f"Correlation heatmap saved to {filepath}")
        return filepath

    def plot_trade_analysis(
        self,
        trade_log: List[dict],
        title: str = "Trade Analysis",
        filename: str = "trade_analysis.png",
    ) -> str:
        """
        Generate trade analysis charts including PnL distribution
        and cumulative PnL curve.

        Args:
            trade_log: List of trade result dictionaries
            title: Chart title
            filename: Output filename

        Returns:
            Path to the saved chart image
        """
        if not trade_log:
            logger.warning("No trades to analyze")
            return ""

        pnls = [t["pnl"] for t in trade_log if "pnl" in t]
        if not pnls:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=self.colors["bg"])

        for ax in axes.flat:
            ax.set_facecolor(self.colors["bg"])

        # 1. PnL Distribution Histogram
        ax1 = axes[0, 0]
        colors = [self.colors["positive"] if p > 0 else self.colors["negative"] for p in pnls]
        ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax1.axhline(y=0, color="white", linewidth=0.5)
        ax1.set_title("Trade PnL", fontsize=12, color="white")
        ax1.set_xlabel("Trade #", color="#888")
        ax1.set_ylabel("PnL ($)", color="#888")
        ax1.grid(True, alpha=0.2)

        # 2. Cumulative PnL
        ax2 = axes[0, 1]
        cum_pnl = np.cumsum(pnls)
        ax2.plot(cum_pnl, color=self.colors["positive"], linewidth=1.5)
        ax2.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                         where=(np.array(cum_pnl) >= 0), alpha=0.15, color=self.colors["positive"])
        ax2.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                         where=(np.array(cum_pnl) < 0), alpha=0.15, color=self.colors["negative"])
        ax2.axhline(y=0, color="white", linewidth=0.5)
        ax2.set_title("Cumulative PnL", fontsize=12, color="white")
        ax2.set_xlabel("Trade #", color="#888")
        ax2.set_ylabel("Cumulative PnL ($)", color="#888")
        ax2.grid(True, alpha=0.2)

        # 3. PnL Distribution
        ax3 = axes[1, 0]
        ax3.hist(pnls, bins=30, color=self.colors["primary"], alpha=0.7, edgecolor="white", linewidth=0.5)
        ax3.axvline(x=0, color="white", linewidth=1)
        ax3.axvline(x=np.mean(pnls), color=self.colors["positive"], linewidth=1.5, linestyle="--", label=f"Mean: ${np.mean(pnls):.2f}")
        ax3.set_title("PnL Distribution", fontsize=12, color="white")
        ax3.set_xlabel("PnL ($)", color="#888")
        ax3.set_ylabel("Frequency", color="#888")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.2)

        # 4. Win/Loss Pie Chart
        ax4 = axes[1, 1]
        wins = len([p for p in pnls if p > 0])
        losses = len([p for p in pnls if p < 0])
        breakeven = len([p for p in pnls if p == 0])
        sizes = [wins, losses, breakeven]
        labels_list = [f"Wins ({wins})", f"Losses ({losses})", f"Breakeven ({breakeven})"]
        pie_colors = [self.colors["positive"], self.colors["negative"], self.colors["neutral"]]
        # Only include non-zero slices
        filtered = [(s, l, c) for s, l, c in zip(sizes, labels_list, pie_colors) if s > 0]
        if filtered:
            sizes_f, labels_f, colors_f = zip(*filtered)
            ax4.pie(sizes_f, labels=labels_f, colors=colors_f, autopct="%1.1f%%",
                    textprops={"color": "white", "fontsize": 10})
        ax4.set_title("Win/Loss Ratio", fontsize=12, color="white")

        fig.suptitle(title, fontsize=16, color="white", y=1.02)
        fig.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)

        logger.info(f"Trade analysis chart saved to {filepath}")
        return filepath

    def generate_all_charts(self, backtest_results: dict, btc_data=None, gold_data=None) -> Dict[str, str]:
        """
        Generate all visualization charts from backtest results.

        Args:
            backtest_results: Complete backtest result dict
            btc_data: Optional BTC DataFrame with indicators for correlation
            gold_data: Optional Gold DataFrame with indicators for correlation

        Returns:
            Dict of chart_name -> file_path
        """
        charts = {}

        # Equity curve
        equity_data = backtest_results.get("equity_curve", [])
        if equity_data:
            charts["equity_curve"] = self.plot_equity_curve(equity_data)
            charts["drawdown"] = self.plot_drawdown(equity_data)

        # Trade analysis
        trade_log = backtest_results.get("trade_log", [])
        if trade_log:
            charts["trade_analysis"] = self.plot_trade_analysis(trade_log)

        # Correlation heatmap (if both BTC and Gold data available)
        if btc_data is not None and gold_data is not None:
            charts["correlation"] = self.plot_correlation_heatmap(btc_data, gold_data)

        logger.info(f"Generated {len(charts)} charts")
        return charts
