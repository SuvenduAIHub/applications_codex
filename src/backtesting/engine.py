"""
Backtesting engine for historical strategy simulation.
Runs strategies against historical data with realistic assumptions
including slippage, fees, and liquidity constraints.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    BacktestConfig,
    ExecutionConfig,
    OrderSide,
    OrderType,
    RiskConfig,
    StrategyConfig,
)
from src.execution.broker import SimulatedBroker
from src.execution.order import Order
from src.execution.portfolio import Portfolio
from src.indicators.technical import add_all_indicators
from src.indicators.regime import compute_regime_features
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class BacktestEngine:
    """
    Historical backtesting engine with realistic simulation.

    Features:
        - Processes historical OHLCV data bar-by-bar
        - Supports multiple strategies (including ensemble)
        - Includes transaction costs and slippage
        - Tracks full equity curve for performance analysis
        - Enforces risk management rules during simulation
        - Supports walk-forward validation
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        backtest_config: Optional[BacktestConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        """
        Initialize the backtesting engine.

        Args:
            strategy: Strategy instance to backtest
            backtest_config: Backtesting configuration
            risk_config: Risk management configuration
            execution_config: Execution/fill configuration
        """
        self.strategy = strategy
        self.bt_config = backtest_config or BacktestConfig()
        self.risk_config = risk_config or RiskConfig()
        self.exec_config = execution_config or ExecutionConfig()

        # These are initialized in run()
        self.portfolio: Optional[Portfolio] = None
        self.broker: Optional[SimulatedBroker] = None
        self.risk_manager: Optional[RiskManager] = None

        # Results storage
        self.equity_history: List[dict] = []
        self.trade_log: List[dict] = []
        self.signal_log: List[dict] = []
        self._initial_balance: float = self.bt_config.initial_balance_usd

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        initial_balance: Optional[float] = None,
    ) -> dict:
        """
        Run a full backtest on historical data.

        Args:
            data: Dict of symbol -> OHLCV DataFrame (must have DatetimeIndex)
                  e.g., {"BTC/USDT": btc_df, "XAU/USD": gold_df}
            initial_balance: Starting portfolio balance (overrides config)

        Returns:
            Dict with complete backtest results and metrics
        """
        balance = initial_balance or self.bt_config.initial_balance_usd
        self._initial_balance = balance

        # Initialize components
        self.portfolio = Portfolio(initial_balance=balance)
        self.broker = SimulatedBroker(self.portfolio, self.exec_config)
        self.risk_manager = RiskManager(self.risk_config)
        self.risk_manager.initialize(balance)
        self.strategy.reset()

        self.equity_history = []
        self.trade_log = []
        self.signal_log = []

        # Enrich data with indicators (avoid duplicating if already enriched)
        enriched_data = {}
        for symbol, df in data.items():
            enriched = add_all_indicators(df)
            enriched = compute_regime_features(enriched)
            # Remove any duplicate columns to prevent Series-as-scalar issues
            enriched = enriched.loc[:, ~enriched.columns.duplicated()]
            enriched_data[symbol] = enriched

        # Get the common time range across all symbols
        all_indices = [df.index for df in enriched_data.values()]
        common_start = max(idx.min() for idx in all_indices)
        common_end = min(idx.max() for idx in all_indices)

        logger.info(
            f"Starting backtest: {common_start} to {common_end}, "
            f"initial balance=${balance:,.2f}"
        )

        # Iterate through each time bar
        bar_count = 0
        for symbol, df in enriched_data.items():
            # Filter to common range
            mask = (df.index >= common_start) & (df.index <= common_end)
            enriched_data[symbol] = df[mask]

        # Use the first symbol's index as the time driver
        primary_symbol = list(enriched_data.keys())[0]
        time_index = enriched_data[primary_symbol].index

        # Minimum lookback before generating signals (to ensure indicator stability)
        warmup_bars = 60

        for i, timestamp in enumerate(time_index):
            if i < warmup_bars:
                continue

            bar_count += 1

            # Process each symbol at this timestamp
            for symbol, df in enriched_data.items():
                if timestamp not in df.index:
                    continue

                # Get data up to (and including) current bar
                current_data = df.loc[:timestamp]
                current_bar = df.loc[timestamp]
                current_price = current_bar["close"]
                current_high = current_bar["high"]
                current_low = current_bar["low"]

                # 1. Process pending orders with current tick
                filled_orders = self.broker.process_tick(
                    symbol, current_price, current_high, current_low
                )

                # Handle filled orders -> update portfolio
                for order in filled_orders:
                    if order.side == OrderSide.BUY:
                        self.portfolio.open_position(
                            symbol=symbol,
                            side="buy",
                            quantity=order.filled_quantity,
                            price=order.filled_price,
                            commission=order.commission,
                        )
                        self.risk_manager.register_position(
                            symbol=symbol,
                            side="buy",
                            size_usd=order.filled_quantity * order.filled_price,
                            entry_price=order.filled_price,
                            stop_loss=order.stop_price or (order.filled_price * 0.97),
                            take_profit=order.filled_price * 1.06,
                        )
                    else:
                        result = self.portfolio.close_position(
                            symbol=symbol,
                            price=order.filled_price,
                            commission=order.commission,
                            reason="signal",
                        )
                        if result:
                            self.trade_log.append(result)
                            self.risk_manager.close_position(symbol, order.filled_price)

                # 2. Check stop-loss / take-profit for open positions
                if symbol in self.portfolio.positions:
                    trigger = self.risk_manager.check_stop_levels(symbol, current_price)
                    if trigger:
                        result = self.portfolio.close_position(
                            symbol=symbol,
                            price=current_price,
                            commission=self.exec_config.simulated_commission_pct / 100 * current_price * self.portfolio.positions.get(symbol, type("", (), {"quantity": 0})).quantity if symbol in self.portfolio.positions else 0,
                            reason=trigger,
                        )
                        if result:
                            self.trade_log.append(result)
                            self.risk_manager.close_position(symbol, current_price)
                            self.broker.cancel_all_orders(symbol)

                    # Update trailing stops
                    self.risk_manager.update_trailing_stop(symbol, current_price)

                # 3. Generate strategy signal
                self.risk_manager.update_portfolio_value(self.portfolio.total_value)
                if self.strategy.should_trade():
                    signal = self.strategy.generate_signal(current_data, symbol)
                    self.signal_log.append({
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        "symbol": symbol,
                        "signal": signal.signal.value,
                        "confidence": signal.confidence,
                        "price": current_price,
                    })

                    # 4. Execute signal if approved by risk manager
                    # Only trade signals with sufficient confidence (reduces false signals)
                    if signal.is_buy and symbol not in self.portfolio.positions and signal.confidence >= 0.55:
                        # Calculate position size using master sizer (applies exposure cap)
                        size_usd = self.risk_manager.position_sizer.calculate_position_size(
                            method="volatility",
                            portfolio_value=self.portfolio.total_value,
                            current_price=current_price,
                            atr=current_bar.get("atr", current_price * 0.02),
                        )
                        can_trade, reason = self.risk_manager.can_trade(symbol, "buy", size_usd)
                        if can_trade:
                            quantity = size_usd / current_price
                            order = Order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=quantity,
                                strategy=self.strategy.get_name(),
                            )
                            self.broker.submit_order(order)
                            self.strategy.on_trade_executed()

                    elif signal.is_sell and symbol in self.portfolio.positions and signal.confidence >= 0.50:
                        pos = self.portfolio.positions[symbol]
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=pos.quantity,
                            strategy=self.strategy.get_name(),
                        )
                        self.broker.submit_order(order)
                        self.strategy.on_trade_executed()

                # Update portfolio prices
                self.portfolio.update_prices({symbol: current_price})

            # Record equity at each bar
            self.equity_history.append({
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "equity": self.portfolio.total_value,
                "cash": self.portfolio.cash_balance,
            })

        # Close any remaining open positions at the last price
        for symbol in list(self.portfolio.positions.keys()):
            if symbol in enriched_data:
                last_price = enriched_data[symbol].iloc[-1]["close"]
                result = self.portfolio.close_position(symbol, last_price, reason="backtest_end")
                if result:
                    self.trade_log.append(result)

        logger.info(f"Backtest complete: {bar_count} bars processed")
        return self._compile_results()

    def _compile_results(self) -> dict:
        """
        Compile comprehensive backtest results with all performance metrics.

        Returns:
            Dict with metrics, equity curve, trade log, etc.
        """
        from src.backtesting.performance import PerformanceAnalyzer

        equity_series = pd.Series(
            [e["equity"] for e in self.equity_history],
            index=pd.to_datetime([e["timestamp"] for e in self.equity_history]),
        )

        # Use the performance analyzer for detailed metrics
        analyzer = PerformanceAnalyzer(
            equity_curve=equity_series,
            trade_log=self.trade_log,
            initial_balance=self._initial_balance,
        )
        metrics = analyzer.calculate_all_metrics()

        return {
            "metrics": metrics,
            "equity_curve": self.equity_history,
            "trade_log": self.trade_log,
            "signal_log": self.signal_log,
            "execution_stats": self.broker.get_execution_stats() if self.broker else {},
            "portfolio_summary": self.portfolio.get_summary() if self.portfolio else {},
        }
