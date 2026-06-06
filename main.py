"""
Main entry point for the Automated Trading System.
Supports three operating modes: live trading, paper trading, and backtesting.
Orchestrates all system components: data feeds, strategies, risk management,
execution, monitoring, and visualization.

Usage:
    python main.py --mode paper          # Paper trading with simulated execution
    python main.py --mode backtest       # Run backtesting on historical data
    python main.py --mode live           # Live trading (requires exchange API keys)
    python main.py --mode backtest --visualize  # Backtest with chart generation
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file so exchange API keys and other settings are available
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from config.settings import (
    BacktestConfig,
    ExecutionConfig,
    OrderSide,
    OrderType,
    PositionSide,
    RiskConfig,
    StrategyConfig,
    TimeFrame,
    TradingMode,
    TradingSystemConfig,
    load_config,
)
from src.data.btc_feed import BinanceBTCFeed
from src.data.gold_feed import YahooGoldFeed
from src.data.data_store import DataStore
from src.indicators.technical import add_all_indicators
from src.indicators.regime import compute_regime_features, detect_macro_regime
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.momentum import MomentumCrossoverStrategy
from src.strategies.volatility import VolatilityStrategy
from src.strategies.correlation_macro import CorrelationMacroStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.quality_trend import QualityTrendStrategy
from src.strategies.ml_strategy import XGBoostStrategy, LSTMStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.execution.broker import SimulatedBroker
from src.execution.order import Order
from src.execution.portfolio import Portfolio
from src.risk.risk_manager import RiskManager
from src.backtesting.engine import BacktestEngine
from src.backtesting.performance import PerformanceAnalyzer
from src.backtesting.optimizer import StrategyOptimizer
from src.monitoring.dashboard import TradingDashboard
from src.monitoring.alerts import AlertManager, AlertLevel, AlertType
from src.monitoring.logger_config import setup_logging, TradeLogger, FailureRecovery
from src.visualization.charts import TradingVisualizer


# Global flag for graceful shutdown
_running = True


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _running
    logger.info("Shutdown signal received, stopping gracefully...")
    _running = False


def is_market_open(symbol: str) -> bool:
    """
    Check if the market for the given symbol is currently open.
    BTC trades 24/7. Gold (XAU/USD) follows COMEX/forex hours:
      - Opens Sunday 6:00 PM ET, closes Friday 5:00 PM ET
      - Daily maintenance break 5:00 PM - 6:00 PM ET (Mon-Thu)
    Returns True if the market is open and trades can be executed.
    """
    # BTC/crypto trades 24/7 — always open
    if "BTC" in symbol or "ETH" in symbol or "USDT" in symbol:
        return True

    # Gold (XAU/USD) and other forex/commodity markets follow COMEX hours
    now_et = datetime.now(ZoneInfo("America/New_York"))
    weekday = now_et.weekday()  # 0=Monday, 6=Sunday
    hour = now_et.hour
    minute = now_et.minute

    # Saturday: market fully closed
    if weekday == 5:
        logger.debug(f"{symbol} market closed: Saturday")
        return False

    # Sunday: opens at 6:00 PM ET
    if weekday == 6:
        if hour < 18:
            logger.debug(f"{symbol} market closed: Sunday before 6 PM ET")
            return False
        return True

    # Friday: closes at 5:00 PM ET
    if weekday == 4:
        if hour >= 17:
            logger.debug(f"{symbol} market closed: Friday after 5 PM ET")
            return False
        return True

    # Monday-Thursday: daily maintenance break 5:00 PM - 6:00 PM ET
    if 0 <= weekday <= 3:
        if hour == 17:
            logger.debug(f"{symbol} market closed: daily maintenance break (5-6 PM ET)")
            return False
        return True

    return True


def run_backtest(config: TradingSystemConfig, visualize: bool = False) -> dict:
    """
    Execute a full backtest cycle on historical data.

    Steps:
        1. Fetch historical data for BTC/USDT and XAU/USD
        2. Initialize strategy ensemble
        3. Run backtesting engine
        4. Print performance report
        5. Optionally generate visualization charts

    Args:
        config: Trading system configuration
        visualize: Whether to generate chart visualizations

    Returns:
        Backtest results dictionary
    """
    logger.info("="*60)
    logger.info("  BACKTESTING MODE")
    logger.info("="*60)

    # 1. Fetch historical data
    logger.info("Fetching historical data...")
    btc_feed = BinanceBTCFeed(config.data_feed)
    gold_feed = YahooGoldFeed(config.data_feed)

    btc_data = btc_feed.fetch_historical(
        timeframe=TimeFrame.H1,
        limit=config.data_feed.max_candles_per_request,
    )
    gold_data = gold_feed.fetch_historical(
        timeframe=TimeFrame.H1,
    )

    logger.info(f"BTC data: {len(btc_data)} candles from {btc_data.index.min()} to {btc_data.index.max()}")
    logger.info(f"Gold data: {len(gold_data)} candles from {gold_data.index.min()} to {gold_data.index.max()}")

    # 2. Initialize strategies
    strategy_config = config.strategy

    # Create individual strategies — full professional stack (8 strategies)
    trend_strategy = TrendFollowingStrategy(strategy_config)
    mean_rev_strategy = MeanReversionStrategy(strategy_config)
    breakout_strategy = BreakoutStrategy(strategy_config)
    momentum_strategy = MomentumCrossoverStrategy(strategy_config)
    volatility_strategy = VolatilityStrategy(strategy_config)
    macro_strategy = CorrelationMacroStrategy(strategy_config)
    grid_strategy = GridTradingStrategy(strategy_config)
    quality_strategy = QualityTrendStrategy(strategy_config)
    xgb_strategy = XGBoostStrategy(strategy_config)

    # Train ML strategy on BTC data
    logger.info("Training ML models...")
    btc_enriched = add_all_indicators(btc_data, strategy_config)
    btc_enriched = compute_regime_features(btc_enriched)
    xgb_train_result = xgb_strategy.train(btc_enriched)
    logger.info(f"XGBoost training: {xgb_train_result}")

    # Create ensemble strategy — "best signal" mode with 8 strategies
    ensemble = EnsembleStrategy(
        strategies=[
            trend_strategy, mean_rev_strategy, breakout_strategy,
            momentum_strategy, volatility_strategy, macro_strategy,
            grid_strategy, quality_strategy, xgb_strategy,
        ],
        config=strategy_config,
        min_consensus=0.3,
    )

    # 3. Run backtest
    data = {"BTC/USDT": btc_data, "XAU/USD": gold_data}

    engine = BacktestEngine(
        strategy=ensemble,
        backtest_config=config.backtest,
        risk_config=config.risk,
        execution_config=config.execution,
    )

    results = engine.run(data, config.backtest.initial_balance_usd)

    # 4. Print performance report
    equity_series = pd.Series(
        [e["equity"] for e in results["equity_curve"]],
        index=pd.to_datetime([e["timestamp"] for e in results["equity_curve"]]),
    )
    analyzer = PerformanceAnalyzer(
        equity_curve=equity_series,
        trade_log=results["trade_log"],
        initial_balance=config.backtest.initial_balance_usd,
    )
    report = analyzer.generate_report()
    print(report)

    # 5. Strategy comparison
    logger.info("Running strategy comparison...")
    optimizer = StrategyOptimizer(config.backtest)
    comparison_strategies = {
        "trend_following": TrendFollowingStrategy(strategy_config),
        "mean_reversion": MeanReversionStrategy(strategy_config),
        "breakout": BreakoutStrategy(strategy_config),
        "momentum_crossover": MomentumCrossoverStrategy(strategy_config),
        "volatility": VolatilityStrategy(strategy_config),
        "correlation_macro": CorrelationMacroStrategy(strategy_config),
        "grid_trading": GridTradingStrategy(strategy_config),
        "quality_trend": QualityTrendStrategy(strategy_config),
        "ensemble": EnsembleStrategy(
            strategies=[
                TrendFollowingStrategy(strategy_config),
                MeanReversionStrategy(strategy_config),
                BreakoutStrategy(strategy_config),
                MomentumCrossoverStrategy(strategy_config),
                VolatilityStrategy(strategy_config),
                CorrelationMacroStrategy(strategy_config),
                GridTradingStrategy(strategy_config),
                QualityTrendStrategy(strategy_config),
            ],
            config=strategy_config,
        ),
    }
    comparison_df = optimizer.compare_strategies(comparison_strategies, data)
    print("\n--- Strategy Comparison ---")
    print(comparison_df.to_string(index=False))

    # 6. Generate visualizations
    if visualize:
        logger.info("Generating visualizations...")
        visualizer = TradingVisualizer()

        # Enrich data for correlation heatmap
        btc_enriched = add_all_indicators(btc_data, strategy_config)
        gold_enriched = add_all_indicators(gold_data, strategy_config)

        charts = visualizer.generate_all_charts(
            results,
            btc_data=btc_enriched,
            gold_data=gold_enriched,
        )
        print(f"\nCharts saved to: {visualizer.output_dir}")
        for name, path in charts.items():
            print(f"  - {name}: {path}")

    # 7. Print execution statistics
    exec_stats = results.get("execution_stats", {})
    print("\n--- Execution Statistics ---")
    for key, value in exec_stats.items():
        print(f"  {key}: {value}")

    return results


def run_paper_trading(config: TradingSystemConfig):
    """
    Run the paper trading loop with simulated execution.
    Continuously fetches live prices, generates signals, and simulates trades.

    Args:
        config: Trading system configuration
    """
    global _running

    logger.info("="*60)
    logger.info("  PAPER TRADING MODE")
    logger.info("="*60)

    # Initialize components
    portfolio = Portfolio(initial_balance=config.backtest.initial_balance_usd)
    broker = SimulatedBroker(portfolio, config.execution)
    risk_manager = RiskManager(config.risk)
    risk_manager.initialize(portfolio.total_value)
    alert_manager = AlertManager()

    # Initialize data feeds
    btc_feed = BinanceBTCFeed(config.data_feed)
    gold_feed = YahooGoldFeed(config.data_feed)

    # Initialize strategies — full professional stack with 7 strategies
    strategy_config = config.strategy
    ensemble = EnsembleStrategy(
        strategies=[
            TrendFollowingStrategy(strategy_config),
            MeanReversionStrategy(strategy_config),
            BreakoutStrategy(strategy_config),
            MomentumCrossoverStrategy(strategy_config),
            VolatilityStrategy(strategy_config),
            CorrelationMacroStrategy(strategy_config),
            GridTradingStrategy(strategy_config),
            QualityTrendStrategy(strategy_config),
        ],
        config=strategy_config,
    )

    # Initialize currency converter for dual-currency display (INR + USD)
    from config.currency import CurrencyConverter
    converter = CurrencyConverter()
    base_currency = config.currency.base_currency.value

    # Start monitoring dashboard with currency support — exchange is "paper" for simulated mode
    dashboard = TradingDashboard(port=config.monitoring.dashboard_port)
    dashboard.set_components(
        portfolio, risk_manager, ensemble, "paper",
        currency_converter=converter, base_currency=base_currency,
        exchange_name="paper", leverage=config.execution.leverage,
    )
    dashboard.start(threaded=True)
    logger.info(f"Dashboard running at http://localhost:{config.monitoring.dashboard_port}")

    # Recovery system
    recovery = FailureRecovery()
    trade_logger = TradeLogger()

    logger.info("Paper trading loop started. Press Ctrl+C to stop.")

    iteration = 0
    while _running:
        try:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")

            # Fetch latest data for each pair
            for symbol, feed in [("BTC/USDT", btc_feed), ("XAU/USD", gold_feed)]:
                try:
                    # Skip this symbol if its market is closed
                    market_open = is_market_open(symbol)

                    # Get recent candles for indicator calculation
                    # Use M5 (5-minute) candles for intraday scalping — generates signals
                    # 12x faster than H1 and allows the system to react to short-term moves
                    df = feed.fetch_historical(timeframe=TimeFrame.M5, limit=200)
                    if df.empty:
                        continue

                    # Enrich with indicators
                    enriched = add_all_indicators(df, strategy_config)
                    enriched = compute_regime_features(enriched)

                    current_price = df.iloc[-1]["close"]
                    logger.info(f"{symbol} price: ${current_price:,.2f}")
                    portfolio.update_prices({symbol: current_price})
                    risk_manager.update_portfolio_value(portfolio.total_value)

                    # Feed enriched candle data to dashboard for TradingView chart
                    dashboard.update_candle_data(symbol, enriched)

                    if not market_open:
                        logger.info(f"{symbol} market is CLOSED — price/dashboard updated, skipping new trades")
                        continue

                    # Process pending orders
                    filled = broker.process_tick(
                        symbol, current_price,
                        df.iloc[-1]["high"], df.iloc[-1]["low"]
                    )
                    for order in filled:
                        trade_logger.log_order_filled(
                            order.order_id, symbol, order.side.value,
                            order.filled_quantity, order.filled_price, order.commission
                        )

                    # Check stop levels for open positions
                    position_closed_this_tick = False
                    if symbol in portfolio.positions:
                        trigger = risk_manager.check_stop_levels(symbol, current_price)
                        if trigger:
                            result = portfolio.close_position(symbol, current_price, reason=trigger)
                            if result:
                                position_closed_this_tick = True
                                trade_logger.log_position_closed(
                                    symbol, result["pnl"], result["pnl_pct"], trigger
                                )
                                risk_manager.close_position(symbol, current_price)
                                alert_manager.send_alert(
                                    AlertType.STOP_LOSS_HIT if trigger == "stop_loss" else AlertType.TAKE_PROFIT_HIT,
                                    AlertLevel.INFO,
                                    f"{trigger.upper()} hit for {symbol}: PnL=${result['pnl']:,.2f}"
                                )

                        new_stop = risk_manager.update_trailing_stop(symbol, current_price)
                        if new_stop and symbol in portfolio.positions:
                            portfolio.positions[symbol].stop_loss = new_stop

                    if position_closed_this_tick:
                        logger.info(f"{symbol} position closed by risk rule; skipping new entry until next tick")
                        continue

                    # Generate signal — always check, no cooldown gating
                    signal = ensemble.generate_signal(enriched, symbol)
                    trade_logger.log_signal(
                        symbol, signal.signal.value, signal.confidence,
                        signal.strategy_name, current_price
                    )

                    atr = enriched.iloc[-1].get("atr", current_price * 0.02)
                    has_position = symbol in portfolio.positions
                    # Check position direction: "buy" for long, "sell" for short
                    current_side = None
                    if has_position:
                        pos_side = portfolio.positions[symbol].side
                        current_side = "buy" if pos_side == PositionSide.LONG else "sell"

                    # Existing positions are managed only by stop-loss, take-profit, and trailing stop.
                    # Opposite strategy signals are ignored until the current position exits by risk rules.
                    if has_position:
                        logger.info(
                            f"{symbol} already has an open {current_side.upper()} position; "
                            "ignoring new direction signals until stop/trailing/take-profit exit"
                        )

                    # SELL signal + no position → open short directly
                    elif signal.is_sell and signal.confidence >= 0.62:
                        base_size_usd = risk_manager.position_sizer.calculate_position_size(
                            method="volatility",
                            portfolio_value=portfolio.total_value,
                            current_price=current_price,
                            atr=atr,
                        )
                        size_usd = base_size_usd * config.execution.leverage
                        can_trade, reason = risk_manager.can_trade(symbol, "sell", base_size_usd)
                        if can_trade:
                            quantity = size_usd / current_price
                            stop_loss = (
                                risk_manager.calculate_stop_loss(
                                    current_price, "sell", atr=atr, position_notional_usd=size_usd
                                )
                                if config.risk.fixed_stop_loss_usd > 0 else signal.stop_loss
                            )
                            if config.risk.fixed_take_profit_usd > 0:
                                take_profit = current_price - config.risk.fixed_take_profit_usd
                            elif config.risk.fixed_stop_loss_usd > 0 or config.risk.trailing_stop_activation_usd > 0:
                                take_profit = None
                            else:
                                take_profit = signal.take_profit or risk_manager.calculate_take_profit(
                                    current_price, "sell", stop_loss or (current_price + 2.5 * atr)
                                )
                            portfolio.open_position(
                                symbol=symbol, side="sell", quantity=quantity,
                                price=current_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                            )
                            risk_manager.register_position(
                                symbol=symbol, side="sell", size_usd=base_size_usd,
                                entry_price=current_price,
                                stop_loss=stop_loss or (current_price + 2.5 * atr),
                                take_profit=take_profit,
                                notional_usd=size_usd,
                            )
                            ensemble.on_trade_executed()
                            dashboard.add_trade_marker(symbol, "sell", current_price)
                            sl_str = f"${stop_loss:,.2f}" if stop_loss else "N/A"
                            tp_str = f"${take_profit:,.2f}" if take_profit else "N/A"
                            logger.info(
                                f"SHORT OPENED: {symbol} qty={quantity:.6f} @ ${current_price:,.2f} "
                                f"= ${size_usd:,.2f} | margin=${base_size_usd:,.2f} "
                                f"| leverage={config.execution.leverage}x | SL={sl_str} TP={tp_str}"
                            )
                        else:
                            logger.info(f"Trade blocked: {reason}")

                    # BUY signal + no position → open long directly
                    elif signal.is_buy and signal.confidence >= 0.62:
                        base_size_usd = risk_manager.position_sizer.calculate_position_size(
                            method="volatility",
                            portfolio_value=portfolio.total_value,
                            current_price=current_price,
                            atr=atr,
                        )
                        size_usd = base_size_usd * config.execution.leverage
                        can_trade, reason = risk_manager.can_trade(symbol, "buy", base_size_usd)
                        if can_trade:
                            quantity = size_usd / current_price
                            stop_loss = (
                                risk_manager.calculate_stop_loss(
                                    current_price, "buy", atr=atr, position_notional_usd=size_usd
                                )
                                if config.risk.fixed_stop_loss_usd > 0 else signal.stop_loss
                            )
                            if config.risk.fixed_take_profit_usd > 0:
                                take_profit = current_price + config.risk.fixed_take_profit_usd
                            elif config.risk.fixed_stop_loss_usd > 0 or config.risk.trailing_stop_activation_usd > 0:
                                take_profit = None
                            else:
                                take_profit = signal.take_profit or risk_manager.calculate_take_profit(
                                    current_price, "buy", stop_loss or (current_price - 2.5 * atr)
                                )
                            portfolio.open_position(
                                symbol=symbol, side="buy", quantity=quantity,
                                price=current_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                            )
                            risk_manager.register_position(
                                symbol=symbol, side="buy", size_usd=base_size_usd,
                                entry_price=current_price,
                                stop_loss=stop_loss or (current_price - 2.5 * atr),
                                take_profit=take_profit,
                                notional_usd=size_usd,
                            )
                            ensemble.on_trade_executed()
                            dashboard.add_trade_marker(symbol, "buy", current_price)
                            sl_str = f"${stop_loss:,.2f}" if stop_loss else "N/A"
                            tp_str = f"${take_profit:,.2f}" if take_profit else "N/A"
                            logger.info(
                                f"LONG OPENED: {symbol} qty={quantity:.6f} @ ${current_price:,.2f} "
                                f"= ${size_usd:,.2f} | margin=${base_size_usd:,.2f} "
                                f"| leverage={config.execution.leverage}x | SL={sl_str} TP={tp_str}"
                            )
                        else:
                            logger.info(f"Trade blocked: {reason}")

                    # Update portfolio prices
                    portfolio.update_prices({symbol: current_price})

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    alert_manager.send_alert(
                        AlertType.SYSTEM_ERROR, AlertLevel.WARNING,
                        f"Error processing {symbol}: {e}"
                    )

            # Record equity
            portfolio.record_equity()

            # Log portfolio summary
            summary = portfolio.get_summary()
            logger.info(
                f"Portfolio: ${summary['total_value']:,.2f} "
                f"(return: {summary['total_return_pct']:.2f}%, "
                f"positions: {len(summary['open_positions'])})"
            )

            # Check risk alerts
            risk_summary = risk_manager.get_risk_summary()
            if risk_summary["trading_halted"]:
                alert_manager.send_alert(
                    AlertType.TRADING_HALTED, AlertLevel.CRITICAL,
                    f"Trading halted: {risk_summary['halt_reason']}"
                )

            # Save state periodically for recovery
            if iteration % 10 == 0:
                recovery.save_state({
                    "portfolio": summary,
                    "risk": risk_summary,
                    "iteration": iteration,
                })

            # Wait for next polling interval
            logger.info(f"Sleeping {config.data_feed.polling_interval_seconds}s until next iteration...")
            for _ in range(config.data_feed.polling_interval_seconds):
                if not _running:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Paper trading loop error: {e}")
            alert_manager.send_alert(
                AlertType.SYSTEM_ERROR, AlertLevel.CRITICAL,
                f"Trading loop error: {e}"
            )
            time.sleep(10)  # Brief cooldown before retry

    # Graceful shutdown
    logger.info("Shutting down paper trading...")
    # Close all positions
    for symbol in list(portfolio.positions.keys()):
        try:
            price_data = btc_feed.fetch_current_price() if "BTC" in symbol else gold_feed.fetch_current_price()
            portfolio.close_position(symbol, price_data["price"], reason="shutdown")
        except Exception:
            pass

    # Final summary
    print("\n" + "="*60)
    print("  PAPER TRADING SESSION SUMMARY")
    print("="*60)
    final_summary = portfolio.get_summary()
    print(f"  Final Value:     ${final_summary['total_value']:,.2f}")
    print(f"  Total Return:    {final_summary['total_return_pct']:.2f}%")
    print(f"  Total Trades:    {final_summary['total_trades']}")
    print(f"  Realized PnL:    ${final_summary['total_realized_pnl']:,.2f}")
    print("="*60)


def run_live_trading(config: TradingSystemConfig, exchange: str = "auto"):
    """
    Run live trading with real exchange execution.
    Supports Binance (USDT), WazirX (INR), Delta Exchange (INR + USDT),
    and XM (Gold XAU/USD + BTC/USD CFDs via MetaTrader 5).

    IMPORTANT: Uses real money. Ensure API keys are set and testnet is used first.

    Args:
        config: Trading system configuration with exchange API credentials
        exchange: Exchange to use — "binance", "wazirx", "delta", "xm", or "auto" (pick by currency)
    """
    global _running
    from config.currency import CurrencyConverter
    from src.execution.live_broker import BinanceLiveBroker, WazirXLiveBroker, DeltaExchangeLiveBroker, XMBroker

    logger.info("=" * 60)
    logger.info("  LIVE TRADING MODE")
    logger.info(f"  Base Currency: {config.currency.base_currency.value}")
    logger.info(f"  Exchange: {exchange}")
    logger.info("=" * 60)

    # Initialize currency converter for dual-currency display
    converter = CurrencyConverter()
    base_currency = config.currency.base_currency.value

    # Initialize portfolio
    portfolio = Portfolio(initial_balance=config.backtest.initial_balance_usd)
    risk_manager = RiskManager(config.risk)
    risk_manager.initialize(portfolio.total_value)
    alert_manager = AlertManager()

    # Auto-detect exchange from currency if not explicitly set
    if exchange == "auto":
        if base_currency == "INR":
            exchange = "wazirx"
        else:
            exchange = "binance"

    # Select and initialize the appropriate exchange broker
    if exchange == "delta":
        # Delta Exchange — supports both INR and USDT
        if not config.execution.delta_api_key or not config.execution.delta_api_secret:
            logger.error("Delta API keys not set! Set DELTA_API_KEY and DELTA_API_SECRET env vars.")
            logger.warning("Falling back to paper trading mode.")
            run_paper_trading(config)
            return
        broker = DeltaExchangeLiveBroker(
            api_key=config.execution.delta_api_key,
            api_secret=config.execution.delta_api_secret,
            portfolio=portfolio,
            testnet=config.execution.delta_testnet,
            currency=base_currency,
        )
        logger.info(f"Using Delta Exchange ({base_currency} trading, testnet={config.execution.delta_testnet})")

    elif exchange == "xm":
        # XM broker via MetaTrader 5 — supports Gold (XAU/USD) and BTC/USD CFDs
        if not config.execution.xm_mt5_login or not config.execution.xm_mt5_password:
            logger.error("XM MT5 credentials not set! Set XM_MT5_LOGIN, XM_MT5_PASSWORD, and XM_MT5_SERVER env vars.")
            logger.warning("Falling back to paper trading mode.")
            run_paper_trading(config)
            return
        broker = XMBroker(
            mt5_login=config.execution.xm_mt5_login,
            mt5_password=config.execution.xm_mt5_password,
            mt5_server=config.execution.xm_mt5_server,
            portfolio=portfolio,
            demo=config.execution.xm_demo,
        )
        logger.info(f"Using XM broker (MT5 server: {config.execution.xm_mt5_server}, demo={config.execution.xm_demo})")

    elif exchange == "wazirx":
        # WazirX for INR-based trading
        if not config.execution.wazirx_api_key or not config.execution.wazirx_api_secret:
            logger.error("WazirX API keys not set! Set WAZIRX_API_KEY and WAZIRX_API_SECRET env vars.")
            logger.warning("Falling back to paper trading mode.")
            run_paper_trading(config)
            return
        broker = WazirXLiveBroker(
            api_key=config.execution.wazirx_api_key,
            api_secret=config.execution.wazirx_api_secret,
            portfolio=portfolio,
        )
        logger.info("Using WazirX exchange (INR trading)")

    else:
        # Binance for USDT-based trading (default)
        if not config.execution.binance_api_key or not config.execution.binance_api_secret:
            logger.error("Binance API keys not set! Set BINANCE_API_KEY and BINANCE_API_SECRET env vars.")
            logger.warning("Falling back to paper trading mode.")
            run_paper_trading(config)
            return
        broker = BinanceLiveBroker(
            api_key=config.execution.binance_api_key,
            api_secret=config.execution.binance_api_secret,
            portfolio=portfolio,
            testnet=config.execution.binance_testnet,
        )
        logger.info(f"Using Binance exchange (USDT trading, testnet={config.execution.binance_testnet})")

    # Verify exchange connectivity by fetching balance
    logger.info("Verifying exchange connection...")
    balances = broker.get_account_balance()
    if balances:
        logger.info(f"Exchange balance: {balances}")
    else:
        logger.warning("Could not fetch exchange balance. Check API credentials.")

    # Initialize data feeds
    btc_feed = BinanceBTCFeed(config.data_feed)
    gold_feed = YahooGoldFeed(config.data_feed)

    # Initialize strategies — full professional stack with 7 strategies
    strategy_config = config.strategy
    ensemble = EnsembleStrategy(
        strategies=[
            TrendFollowingStrategy(strategy_config),
            MeanReversionStrategy(strategy_config),
            BreakoutStrategy(strategy_config),
            MomentumCrossoverStrategy(strategy_config),
            VolatilityStrategy(strategy_config),
            CorrelationMacroStrategy(strategy_config),
            GridTradingStrategy(strategy_config),
            QualityTrendStrategy(strategy_config),
        ],
        config=strategy_config,
    )

    # Leverage setting — only applied in live mode (paper always uses 1x)
    leverage = config.execution.leverage
    logger.info(f"Leverage: {leverage}x (live mode)")

    # Start monitoring dashboard — pass exchange name and leverage so user can see trade settings
    dashboard = TradingDashboard(port=config.monitoring.dashboard_port)
    dashboard.set_components(
        portfolio, risk_manager, ensemble, "live",
        currency_converter=converter, base_currency=base_currency,
        exchange_name=exchange, leverage=leverage,
    )
    dashboard.start(threaded=True)
    logger.info(f"Dashboard at http://localhost:{config.monitoring.dashboard_port}")

    recovery = FailureRecovery()
    trade_logger = TradeLogger()

    logger.info("Live trading loop started. Press Ctrl+C to stop.")
    logger.warning("*** REAL MONEY IS AT RISK ***")

    iteration = 0
    while _running:
        try:
            iteration += 1
            logger.info(f"\n--- Live Iteration {iteration} ---")

            # Fetch data and generate signals (same logic as paper trading)
            for symbol, feed in [("BTC/USDT", btc_feed), ("XAU/USD", gold_feed)]:
                try:
                    # Skip this symbol if its market is closed
                    market_open = is_market_open(symbol)

                    df = feed.fetch_historical(timeframe=TimeFrame.M5, limit=200)
                    if df.empty:
                        continue

                    enriched = add_all_indicators(df, strategy_config)
                    enriched = compute_regime_features(enriched)
                    current_price = df.iloc[-1]["close"]
                    portfolio.update_prices({symbol: current_price})
                    risk_manager.update_portfolio_value(portfolio.total_value)

                    # Feed enriched candle data to dashboard for TradingView chart
                    dashboard.update_candle_data(symbol, enriched)

                    if not market_open:
                        logger.info(f"{symbol} market is CLOSED — price/dashboard updated, skipping new trades")
                        continue

                    # Log price in both currencies
                    inr_price = converter.convert(current_price, "USD", "INR")
                    logger.info(f"{symbol}: ${current_price:,.2f} / Rs.{inr_price:,.2f}")

                    # Check stop levels
                    if symbol in portfolio.positions:
                        trigger = risk_manager.check_stop_levels(symbol, current_price)
                        if trigger:
                            # Execute real close order on exchange
                            pos = portfolio.positions[symbol]
                            close_order = Order(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=pos.quantity,
                                strategy="risk_manager",
                            )
                            broker.submit_order(close_order)
                            result = portfolio.close_position(symbol, current_price, reason=trigger)
                            if result:
                                risk_manager.close_position(symbol, current_price)
                                logger.info(f"{trigger.upper()}: {symbol} PnL=${result['pnl']:,.2f}")

                        risk_manager.update_trailing_stop(symbol, current_price)

                    # Generate and execute signals
                    if ensemble.should_trade():
                        signal = ensemble.generate_signal(enriched, symbol)

                        if signal.is_buy and symbol not in portfolio.positions:
                            atr = enriched.iloc[-1].get("atr", current_price * 0.02)
                            # Use master sizing function which applies exposure cap
                            base_size_usd = risk_manager.position_sizer.calculate_position_size(
                                method="volatility",
                                portfolio_value=portfolio.total_value,
                                current_price=current_price,
                                atr=atr,
                            )
                            # Apply leverage — margin required = base_size, effective position = base * leverage
                            size_usd = base_size_usd * leverage
                            can_trade, reason = risk_manager.can_trade(symbol, "buy", base_size_usd)
                            if can_trade:
                                quantity = size_usd / current_price
                                order = Order(
                                    symbol=symbol,
                                    side=OrderSide.BUY,
                                    order_type=OrderType.MARKET,
                                    quantity=quantity,
                                    strategy=ensemble.get_name(),
                                )
                                # Execute on real exchange
                                broker.submit_order(order)
                                # Track position in portfolio and risk manager
                                portfolio.open_position(
                                    symbol=symbol, side="buy", quantity=quantity,
                                    price=current_price,
                                    stop_loss=signal.stop_loss,
                                    take_profit=signal.take_profit,
                                )
                                risk_manager.register_position(
                                    symbol=symbol, side="buy", size_usd=size_usd,
                                    entry_price=current_price,
                                    stop_loss=signal.stop_loss or (current_price - 2.5 * atr),
                                    take_profit=signal.take_profit or (current_price + 4 * atr),
                                )
                                ensemble.on_trade_executed()
                                # Add buy marker to chart
                                dashboard.add_trade_marker(symbol, "buy", current_price)
                                inr_size = converter.convert(size_usd, "USD", "INR")
                                logger.info(
                                    f"LIVE BUY: {symbol} qty={quantity:.6f} @ ${current_price:,.2f} "
                                    f"= ${size_usd:,.2f} / Rs.{inr_size:,.2f} "
                                    f"(leverage={leverage}x, margin=${base_size_usd:,.2f})"
                                )

                        elif signal.is_sell and symbol in portfolio.positions:
                            pos = portfolio.positions[symbol]
                            order = Order(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=pos.quantity,
                                strategy=ensemble.get_name(),
                            )
                            broker.submit_order(order)
                            result = portfolio.close_position(symbol, current_price, reason="sell_signal")
                            if result:
                                risk_manager.close_position(symbol, current_price)
                                ensemble.on_trade_executed()
                                # Add sell marker to chart
                                dashboard.add_trade_marker(symbol, "sell", current_price)
                                logger.info(
                                    f"LIVE SELL: {symbol} @ ${current_price:,.2f} "
                                    f"| PnL=${result['pnl']:,.2f} ({result['pnl_pct']:.2f}%)"
                                )

                    portfolio.update_prices({symbol: current_price})

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Record equity and log summary
            portfolio.record_equity()
            inr_rate = converter.get_rate("USD", "INR")
            summary = portfolio.get_summary(inr_rate=inr_rate)
            logger.info(
                f"Portfolio: ${summary['total_value']:,.2f} / "
                f"Rs.{summary.get('inr', {}).get('total_value', 0):,.2f}"
            )

            # Save state for recovery
            if iteration % 5 == 0:
                recovery.save_state({
                    "portfolio": summary,
                    "iteration": iteration,
                    "mode": "live",
                    "currency": base_currency,
                })

            # Wait for next cycle
            for _ in range(config.data_feed.polling_interval_seconds):
                if not _running:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Live trading error: {e}")
            time.sleep(10)

    # Shutdown
    logger.info("Shutting down live trading...")
    print("\n" + "=" * 60)
    print("  LIVE TRADING SESSION SUMMARY")
    print("=" * 60)
    inr_rate = converter.get_rate("USD", "INR")
    final = portfolio.get_summary(inr_rate=inr_rate)
    print(f"  Final Value (USD): ${final['total_value']:,.2f}")
    if "inr" in final:
        print(f"  Final Value (INR): Rs.{final['inr']['total_value']:,.2f}")
    print(f"  Total Return:      {final['total_return_pct']:.2f}%")
    print(f"  Total Trades:      {final['total_trades']}")
    print(f"  Exchange Rate:     1 USD = Rs.{inr_rate:.2f}")
    print("=" * 60)


def main():
    """Main entry point with CLI argument parsing."""
    # Need pandas for backtest report
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Automated Trading System for BTC/USDT and XAU/USD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode backtest                              # Run backtest
  python main.py --mode backtest --visualize                  # Backtest with charts
  python main.py --mode paper                                 # Paper trading (USD)
  python main.py --mode paper --currency INR                  # Paper trading (INR display)
  python main.py --mode live --currency USD                   # Live via Binance (USDT)
  python main.py --mode live --currency INR                   # Live via WazirX (INR)
  python main.py --mode live --exchange delta --currency INR  # Live via Delta Exchange (INR)
  python main.py --mode live --exchange delta --currency USD  # Live via Delta Exchange (USDT)
  python main.py --mode live --exchange xm --currency USD     # Live via XM (Gold + BTC CFDs)
  python main.py --mode paper --balance 50000                 # Custom balance
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="backtest",
        choices=["live", "paper", "backtest"],
        help="Operating mode (default: backtest)",
    )
    parser.add_argument(
        "--currency", type=str, default="USD",
        choices=["USD", "INR"],
        help="Base currency for deposits and display (default: USD)",
    )
    parser.add_argument(
        "--exchange", type=str, default="auto",
        choices=["auto", "binance", "wazirx", "delta", "xm"],
        help="Exchange broker for live trading (default: auto = Binance for USD, WazirX for INR, xm for Gold+BTC CFDs)",
    )
    parser.add_argument(
        "--balance", type=float, default=100000.0,
        help="Initial portfolio balance in base currency (default: 100000)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Data timeframe (default: 1h)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization charts after backtesting",
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=5000,
        help="Dashboard web server port (default: 5000)",
    )
    parser.add_argument(
        "--leverage", type=int, default=None,
        help="Leverage multiplier for paper/live position notional (default: LEVERAGE env or 1)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load and customize configuration
    config = load_config()
    config.backtest.initial_balance_usd = args.balance
    config.monitoring.dashboard_port = args.dashboard_port
    if args.leverage is not None:
        config.execution.leverage = max(1, args.leverage)

    # Set base currency from CLI argument
    from config.settings import BaseCurrency
    config.currency.base_currency = BaseCurrency(args.currency)

    # Display currency symbol
    currency_symbol = "Rs." if args.currency == "INR" else "$"

    logger.info(f"Starting Automated Trading System (mode={args.mode})")
    logger.info(f"Trading pairs: BTC/USDT, XAU/USD")
    logger.info(f"Base currency: {args.currency}")
    logger.info(f"Initial balance: {currency_symbol}{args.balance:,.2f}")
    logger.info(f"Leverage: {config.execution.leverage}x")

    if args.mode == "backtest":
        config.mode = TradingMode.BACKTEST
        run_backtest(config, visualize=args.visualize)
    elif args.mode == "paper":
        config.mode = TradingMode.PAPER
        run_paper_trading(config)
    elif args.mode == "live":
        config.mode = TradingMode.LIVE
        run_live_trading(config, exchange=args.exchange)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
