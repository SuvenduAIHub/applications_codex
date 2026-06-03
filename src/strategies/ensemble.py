"""
Ensemble strategy that combines signals from multiple sub-strategies.
Uses "best signal" mode: if ANY strategy generates a buy/sell signal,
the highest-confidence signal is used. This ensures the system trades
actively even when only one strategy detects an opportunity.
Previously required consensus (multiple strategies agreeing), which
was too restrictive and caused 9+ hours of no trades.
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.indicators.regime import MarketRegime
from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy that aggregates signals from multiple sub-strategies.

    Features:
        - "Best signal" mode: uses the highest-confidence signal from any strategy
        - Weighted voting as fallback when multiple strategies agree
        - Adaptive weight adjustment based on market regime
        - Performance-tracking to upweight consistently profitable strategies

    Regime-based weighting:
        - Trending market: upweight trend_following, downweight mean_reversion
        - Ranging market: upweight mean_reversion, downweight trend_following
        - High volatility: upweight breakout, downweight mean_reversion
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        config=None,
        min_consensus: float = 0.5,
    ):
        """
        Initialize the ensemble with a list of sub-strategies.

        Args:
            strategies: List of BaseStrategy instances to combine
            config: Strategy configuration
            min_consensus: Minimum fraction of strategies that must agree (0-1).
                           Note: in "best signal" mode this is a soft threshold
                           — a single high-confidence signal can still trigger a trade.
        """
        super().__init__(config)
        self.strategies = strategies
        self.min_consensus = min_consensus

        # Default equal weights for all strategies
        self.strategy_weights: Dict[str, float] = {
            s.get_name(): 1.0 / len(strategies) for s in strategies
        }

        # Performance tracking for adaptive weighting
        self.strategy_performance: Dict[str, List[float]] = {
            s.get_name(): [] for s in strategies
        }

        # Regime-based weight multipliers
        # Key: (regime, strategy_name) -> multiplier
        self.regime_multipliers: Dict = {
            # Trend-following excels in trends
            (MarketRegime.STRONG_UPTREND, "trend_following"): 1.5,
            (MarketRegime.UPTREND, "trend_following"): 1.3,
            (MarketRegime.DOWNTREND, "trend_following"): 1.3,
            (MarketRegime.STRONG_DOWNTREND, "trend_following"): 1.5,
            # Mean reversion excels in ranging markets
            (MarketRegime.RANGING, "mean_reversion"): 1.5,
            # Mean reversion underperforms in strong trends
            (MarketRegime.STRONG_UPTREND, "mean_reversion"): 0.5,
            (MarketRegime.STRONG_DOWNTREND, "mean_reversion"): 0.5,
            # Breakout excels in high volatility and after ranges
            (MarketRegime.HIGH_VOLATILITY, "breakout"): 1.3,
            (MarketRegime.RANGING, "breakout"): 1.2,
            # ML strategies get moderate weight everywhere
            (MarketRegime.STRONG_UPTREND, "xgboost_ml"): 1.1,
            (MarketRegime.STRONG_DOWNTREND, "xgboost_ml"): 1.1,
            # Momentum crossover works well in all conditions, especially ranging
            (MarketRegime.RANGING, "momentum_crossover"): 1.4,
            (MarketRegime.UPTREND, "momentum_crossover"): 1.2,
            (MarketRegime.DOWNTREND, "momentum_crossover"): 1.2,
            # Volatility strategy excels in high volatility and breakout conditions
            (MarketRegime.HIGH_VOLATILITY, "volatility"): 1.5,
            (MarketRegime.RANGING, "volatility"): 1.3,
            (MarketRegime.STRONG_UPTREND, "volatility"): 1.2,
            (MarketRegime.STRONG_DOWNTREND, "volatility"): 1.2,
            # Correlation/Macro provides macro-level bias — moderate in all regimes
            (MarketRegime.UPTREND, "correlation_macro"): 1.2,
            (MarketRegime.DOWNTREND, "correlation_macro"): 1.2,
            (MarketRegime.RANGING, "correlation_macro"): 1.1,
            # Grid trading excels in ranging markets, poor in trends
            (MarketRegime.RANGING, "grid_trading"): 1.5,
            (MarketRegime.STRONG_UPTREND, "grid_trading"): 0.3,
            (MarketRegime.STRONG_DOWNTREND, "grid_trading"): 0.3,
            (MarketRegime.HIGH_VOLATILITY, "grid_trading"): 0.5,
        }

    def get_name(self) -> str:
        return "ensemble"

    def _get_regime_adjusted_weights(
        self, regime: Optional[MarketRegime]
    ) -> Dict[str, float]:
        """
        Adjust strategy weights based on the current market regime.

        Args:
            regime: Current detected market regime

        Returns:
            Dict of strategy_name -> adjusted weight
        """
        adjusted = {}
        for strategy in self.strategies:
            name = strategy.get_name()
            base_weight = self.strategy_weights[name]
            multiplier = 1.0

            if regime is not None:
                multiplier = self.regime_multipliers.get((regime, name), 1.0)

            adjusted[name] = base_weight * multiplier

        # Normalize weights to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate an ensemble signal by combining sub-strategy signals.

        Args:
            df: DataFrame with all indicators
            symbol: Trading pair symbol

        Returns:
            Weighted ensemble TradeSignal
        """
        price = df.iloc[-1]["close"] if len(df) > 0 else 0.0

        # Detect current regime for adaptive weighting
        current_regime = None
        if "regime" in df.columns:
            last_regime = df.iloc[-1]["regime"]
            if isinstance(last_regime, MarketRegime):
                current_regime = last_regime

        # Get regime-adjusted weights
        weights = self._get_regime_adjusted_weights(current_regime)

        # Collect signals from all sub-strategies
        signals: List[TradeSignal] = []
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(df, symbol)
                signals.append(signal)
            except Exception as e:
                logger.warning(f"Strategy {strategy.get_name()} failed: {e}")

        if not signals:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD, confidence=0.0,
                strategy_name=self.get_name(), price=price,
            )

        # Collect all signals with their metadata
        all_metadata = {}
        buy_signals = []
        sell_signals = []

        for sig in signals:
            weight = weights.get(sig.strategy_name, 0.0)
            all_metadata[sig.strategy_name] = {
                "signal": sig.signal.value,
                "confidence": sig.confidence,
                "weight": weight,
            }
            if sig.is_buy:
                buy_signals.append(sig)
            elif sig.is_sell:
                sell_signals.append(sig)

        # --- "Best Signal" mode ---
        # If ANY strategy generates a signal with confidence > 0.3, use it.
        # This ensures the system trades even when only one strategy fires.
        # When multiple strategies agree, use the highest weighted confidence.
        best_buy = max(buy_signals, key=lambda s: s.confidence, default=None)
        best_sell = max(sell_signals, key=lambda s: s.confidence, default=None)

        # Calculate weighted scores for tie-breaking
        buy_score = sum(s.confidence * weights.get(s.strategy_name, 0.0) for s in buy_signals)
        sell_score = sum(s.confidence * weights.get(s.strategy_name, 0.0) for s in sell_signals)

        total_strategies = len(signals)
        buy_consensus = len(buy_signals) / total_strategies if total_strategies > 0 else 0
        sell_consensus = len(sell_signals) / total_strategies if total_strategies > 0 else 0

        # Minimum confidence threshold — balanced for quality trades over quantity
        # Higher threshold filters out weak/noisy signals that cause whipsaw losses
        min_confidence_threshold = 0.45

        # Determine ensemble signal — best signal wins
        if best_buy and (not best_sell or buy_score >= sell_score):
            if best_buy.confidence >= min_confidence_threshold:
                # Boost confidence if multiple strategies agree
                confidence = min(1.0, best_buy.confidence + (buy_consensus * 0.2))
                signal = Signal.STRONG_BUY if confidence > 0.7 else Signal.BUY

                # Use the best signal's stop/target, or most conservative if multiple
                buy_with_sl = [s for s in buy_signals if s.stop_loss is not None]
                stop_loss = max(s.stop_loss for s in buy_with_sl) if buy_with_sl else best_buy.stop_loss
                take_profit = min(
                    (s.take_profit for s in buy_with_sl if s.take_profit is not None),
                    default=best_buy.take_profit,
                )

                return TradeSignal(
                    symbol=symbol, signal=signal, confidence=confidence,
                    strategy_name=self.get_name(), price=price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    metadata={
                        "sub_signals": all_metadata,
                        "best_strategy": best_buy.strategy_name,
                        "buy_consensus": buy_consensus,
                        "regime": current_regime.value if current_regime else "unknown",
                    },
                )

        if best_sell and (not best_buy or sell_score > buy_score):
            if best_sell.confidence >= min_confidence_threshold:
                confidence = min(1.0, best_sell.confidence + (sell_consensus * 0.2))
                signal = Signal.STRONG_SELL if confidence > 0.7 else Signal.SELL

                sell_with_sl = [s for s in sell_signals if s.stop_loss is not None]
                stop_loss = min(s.stop_loss for s in sell_with_sl) if sell_with_sl else best_sell.stop_loss
                take_profit = max(
                    (s.take_profit for s in sell_with_sl if s.take_profit is not None),
                    default=best_sell.take_profit,
                )

                return TradeSignal(
                    symbol=symbol, signal=signal, confidence=confidence,
                    strategy_name=self.get_name(), price=price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    metadata={
                        "sub_signals": all_metadata,
                        "best_strategy": best_sell.strategy_name,
                        "sell_consensus": sell_consensus,
                        "regime": current_regime.value if current_regime else "unknown",
                    },
                )

        return TradeSignal(
            symbol=symbol, signal=Signal.HOLD, confidence=0.0,
            strategy_name=self.get_name(), price=price,
            metadata={"sub_signals": all_metadata, "no_signal": True},
        )

    def update_performance(self, strategy_name: str, pnl: float) -> None:
        """
        Update performance tracking for a strategy after a trade closes.
        Used for adaptive weight adjustment.

        Args:
            strategy_name: Name of the strategy that produced the trade
            pnl: Profit/loss of the closed trade
        """
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].append(pnl)
            # Recalculate weights based on recent performance
            self._update_weights_from_performance()

    def _update_weights_from_performance(self, lookback: int = 20) -> None:
        """
        Adaptively adjust strategy weights based on recent trade performance.
        Strategies with higher recent win rates get more weight.

        Args:
            lookback: Number of recent trades to consider
        """
        new_weights = {}
        for name, pnls in self.strategy_performance.items():
            recent = pnls[-lookback:] if len(pnls) >= lookback else pnls
            if recent:
                # Win rate as the performance metric
                win_rate = sum(1 for p in recent if p > 0) / len(recent)
                # Minimum weight floor to prevent complete zeroing out
                new_weights[name] = max(0.05, win_rate)
            else:
                new_weights[name] = 1.0 / len(self.strategies)

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            self.strategy_weights = {k: v / total for k, v in new_weights.items()}

        logger.debug(f"Updated ensemble weights: {self.strategy_weights}")
