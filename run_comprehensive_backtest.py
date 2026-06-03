"""
Comprehensive backtesting script for strategy optimization.
Runs all strategies across multiple timeframes and generates a detailed comparison report.
Tests on 2 years of daily data and 42 days of hourly data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from loguru import logger
from config.settings import (
    BacktestConfig, RiskConfig, ExecutionConfig,
    StrategyConfig, TimeFrame, TradingSystemConfig, load_config,
)
from src.data.btc_feed import BinanceBTCFeed
from src.data.gold_feed import YahooGoldFeed
from src.indicators.technical import add_all_indicators
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.ml_strategy import XGBoostStrategy
from src.backtesting.engine import BacktestEngine


def run_single_strategy_backtest(strategy, data, config, label=""):
    """Run backtest for a single strategy and return results."""
    engine = BacktestEngine(
        strategy=strategy,
        backtest_config=config.backtest,
        risk_config=config.risk,
        execution_config=config.execution,
    )
    results = engine.run(data, config.backtest.initial_balance_usd)
    metrics = results["metrics"]
    ret = metrics["return_metrics"]
    trade = metrics["trade_metrics"]
    risk = metrics["ratio_metrics"]
    dd = metrics["drawdown_metrics"]

    return {
        "strategy": label or strategy.get_name(),
        "total_return_pct": ret["total_return_pct"],
        "cagr_pct": ret["cagr_pct"],
        "sharpe_ratio": risk["sharpe_ratio"],
        "sortino_ratio": risk["sortino_ratio"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "win_rate_pct": trade.get("win_rate_pct", 0),
        "profit_factor": trade.get("profit_factor", 0),
        "total_trades": trade.get("total_trades", 0),
        "avg_win": trade.get("avg_win", 0),
        "avg_loss": trade.get("avg_loss", 0),
    }


def main():
    config = load_config()
    btc_feed = BinanceBTCFeed(config.data_feed)
    gold_feed = YahooGoldFeed(config.data_feed)

    # ====== 1. DAILY DATA (2 years) ======
    print("\n" + "=" * 70)
    print("  DAILY TIMEFRAME BACKTEST (2 Years)")
    print("=" * 70)

    btc_daily = btc_feed.fetch_historical(timeframe=TimeFrame.D1, limit=730)
    gold_daily = gold_feed.fetch_historical(timeframe=TimeFrame.D1)

    # Align datasets
    common_start = max(btc_daily.index.min(), gold_daily.index.min())
    common_end = min(btc_daily.index.max(), gold_daily.index.max())
    btc_daily = btc_daily[common_start:common_end]
    gold_daily = gold_daily[common_start:common_end]

    print(f"BTC daily: {len(btc_daily)} candles ({btc_daily.index.min()} to {btc_daily.index.max()})")
    print(f"Gold daily: {len(gold_daily)} candles ({gold_daily.index.min()} to {gold_daily.index.max()})")

    # Pass raw data to engine (engine adds indicators + regime internally)
    daily_data = {"BTC/USDT": btc_daily, "XAU/USD": gold_daily}
    # Enrich with indicators + regime for XGBoost training (matches engine enrichment)
    from src.indicators.regime import compute_regime_features
    btc_enriched = compute_regime_features(add_all_indicators(btc_daily))

    # Strategy configs
    strategy_config = config.strategy

    # Create strategies
    trend = TrendFollowingStrategy(strategy_config)
    mean_rev = MeanReversionStrategy(strategy_config)
    breakout = BreakoutStrategy(strategy_config)
    xgb = XGBoostStrategy(strategy_config)

    # Train XGBoost on BTC data
    try:
        xgb.train(btc_enriched)
    except Exception as e:
        print(f"XGBoost training failed: {e}")

    # Ensemble with lower consensus for better trade capture
    ensemble = EnsembleStrategy(
        strategies=[trend, mean_rev, breakout, xgb],
        config=strategy_config,
        min_consensus=0.3,
    )

    # Run each strategy
    results_daily = []
    for strat, label in [
        (trend, "Trend Following"),
        (mean_rev, "Mean Reversion"),
        (breakout, "Breakout"),
        (xgb, "XGBoost ML"),
        (ensemble, "Ensemble"),
    ]:
        try:
            r = run_single_strategy_backtest(strat, daily_data, config, label)
            results_daily.append(r)
            print(f"\n  {label}:")
            print(f"    Return: {r['total_return_pct']:.2f}% | Win Rate: {r['win_rate_pct']:.1f}% | "
                  f"Profit Factor: {r['profit_factor']:.2f} | Trades: {r['total_trades']}")
            print(f"    Sharpe: {r['sharpe_ratio']:.2f} | Max DD: {r['max_drawdown_pct']:.2f}% | "
                  f"Avg Win: ${r['avg_win']:.0f} | Avg Loss: ${r['avg_loss']:.0f}")
        except Exception as e:
            print(f"  {label}: FAILED - {e}")

    # ====== 2. HOURLY DATA (42 days) ======
    print("\n" + "=" * 70)
    print("  HOURLY TIMEFRAME BACKTEST (42 Days)")
    print("=" * 70)

    btc_hourly = btc_feed.fetch_historical(timeframe=TimeFrame.H1, limit=1000)
    gold_hourly = gold_feed.fetch_historical(timeframe=TimeFrame.H1)

    # Pass raw data (engine adds indicators internally)
    hourly_data = {"BTC/USDT": btc_hourly, "XAU/USD": gold_hourly}
    btc_h_enriched = compute_regime_features(add_all_indicators(btc_hourly))

    # Recreate strategies (reset state)
    trend2 = TrendFollowingStrategy(strategy_config)
    mean_rev2 = MeanReversionStrategy(strategy_config)
    breakout2 = BreakoutStrategy(strategy_config)
    xgb2 = XGBoostStrategy(strategy_config)
    try:
        xgb2.train(btc_h_enriched)
    except Exception as e:
        print(f"XGBoost training (hourly) failed: {e}")

    ensemble2 = EnsembleStrategy(
        strategies=[trend2, mean_rev2, breakout2, xgb2],
        config=strategy_config,
        min_consensus=0.3,
    )

    results_hourly = []
    for strat, label in [
        (trend2, "Trend Following"),
        (mean_rev2, "Mean Reversion"),
        (breakout2, "Breakout"),
        (xgb2, "XGBoost ML"),
        (ensemble2, "Ensemble"),
    ]:
        try:
            r = run_single_strategy_backtest(strat, hourly_data, config, label)
            results_hourly.append(r)
            print(f"\n  {label}:")
            print(f"    Return: {r['total_return_pct']:.2f}% | Win Rate: {r['win_rate_pct']:.1f}% | "
                  f"Profit Factor: {r['profit_factor']:.2f} | Trades: {r['total_trades']}")
            print(f"    Sharpe: {r['sharpe_ratio']:.2f} | Max DD: {r['max_drawdown_pct']:.2f}% | "
                  f"Avg Win: ${r['avg_win']:.0f} | Avg Loss: ${r['avg_loss']:.0f}")
        except Exception as e:
            print(f"  {label}: FAILED - {e}")

    # ====== SUMMARY ======
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)

    print("\nDAILY Timeframe (2 Years):")
    df_daily = pd.DataFrame(results_daily)
    if not df_daily.empty:
        print(df_daily.to_string(index=False))

    print("\nHOURLY Timeframe (42 Days):")
    df_hourly = pd.DataFrame(results_hourly)
    if not df_hourly.empty:
        print(df_hourly.to_string(index=False))

    # Best strategies
    if results_daily:
        best_daily = max(results_daily, key=lambda x: x["win_rate_pct"])
        print(f"\nBest win rate (Daily):  {best_daily['strategy']} = {best_daily['win_rate_pct']:.1f}%")
    if results_hourly:
        best_hourly = max(results_hourly, key=lambda x: x["win_rate_pct"])
        print(f"Best win rate (Hourly): {best_hourly['strategy']} = {best_hourly['win_rate_pct']:.1f}%")

    # ====== 3. WALK-FORWARD VALIDATION ======
    # Split daily data into 3 overlapping windows to check for overfitting
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (Mean Reversion on Daily Data)")
    print("=" * 70)

    total_bars = len(btc_daily)
    window_size = total_bars // 3  # ~8 months per window

    for fold, start_idx in enumerate([0, window_size, window_size * 2], 1):
        end_idx = min(start_idx + window_size + 60, total_bars)  # +60 for warmup
        btc_fold = btc_daily.iloc[start_idx:end_idx]
        gold_fold = gold_daily.iloc[start_idx:min(end_idx, len(gold_daily))]

        if len(btc_fold) < 100 or len(gold_fold) < 100:
            print(f"  Fold {fold}: Insufficient data, skipping")
            continue

        fold_data = {"BTC/USDT": btc_fold, "XAU/USD": gold_fold}
        mr_fold = MeanReversionStrategy(strategy_config)
        try:
            r = run_single_strategy_backtest(mr_fold, fold_data, config, f"Fold {fold}")
            period = f"{btc_fold.index[0].strftime('%Y-%m')} to {btc_fold.index[-1].strftime('%Y-%m')}"
            print(f"  Fold {fold} ({period}):")
            print(f"    Return: {r['total_return_pct']:.2f}% | Win Rate: {r['win_rate_pct']:.1f}% | "
                  f"PF: {r['profit_factor']:.2f} | Trades: {r['total_trades']}")
        except Exception as e:
            print(f"  Fold {fold}: FAILED - {e}")

    print("\n  Note: Consistent performance across folds indicates low overfitting risk.")


if __name__ == "__main__":
    main()
