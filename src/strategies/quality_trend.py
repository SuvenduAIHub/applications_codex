"""
Quality trend strategy.

This strategy is deliberately selective. It trades only when the market has
trend strength, directional momentum, volume confirmation, and price alignment
with VWAP/EMA/Supertrend. It is designed as a capital-preservation filter for
live/paper trading, not as a high-frequency signal generator.
"""

import pandas as pd

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class QualityTrendStrategy(BaseStrategy):
    """Selective trend-continuation strategy for cleaner entries."""

    def get_name(self) -> str:
        return "quality_trend"

    @staticmethod
    def _value(row, name: str, default):
        value = row.get(name, default)
        if hasattr(value, "iloc"):
            return value.iloc[0] if len(value) else default
        return value

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        if len(df) < 60:
            return self._neutral_signal(df, symbol)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(curr["close"])

        ema_fast = float(self._value(curr, "ema_12", self._value(curr, "ema_fast", price)))
        ema_slow = float(self._value(curr, "ema_26", self._value(curr, "ema_slow", price)))
        sma_50 = float(self._value(curr, "sma_50", price))
        vwap = float(self._value(curr, "vwap", price))
        atr = float(self._value(curr, "atr", price * 0.02))
        adx = float(self._value(curr, "adx", 0.0))
        rsi = float(self._value(curr, "rsi", 50.0))
        volume_ratio = float(self._value(curr, "volume_ratio", 1.0))
        macd_hist = float(self._value(curr, "macd_histogram", 0.0))
        prev_macd_hist = float(self._value(prev, "macd_histogram", 0.0))
        supertrend_direction = float(self._value(curr, "supertrend_direction", 0.0))

        if atr <= 0 or price <= 0:
            return self._neutral_signal(df, symbol)

        buy_factors = {}
        sell_factors = {}

        trend_is_tradeable = adx >= 18
        volume_is_tradeable = volume_ratio >= 0.75

        if trend_is_tradeable and volume_is_tradeable:
            if ema_fast > ema_slow and price > sma_50 and price > vwap:
                buy_factors["trend_alignment"] = 0.85
            if macd_hist > 0 and macd_hist >= prev_macd_hist:
                buy_factors["momentum_expanding"] = 0.75
            if 45 <= rsi <= 68:
                buy_factors["rsi_healthy"] = 0.65
            if supertrend_direction == 1:
                buy_factors["supertrend_confirmed"] = 0.75
            if adx >= 25:
                buy_factors["strong_adx"] = 0.7

            if ema_fast < ema_slow and price < sma_50 and price < vwap:
                sell_factors["trend_alignment"] = 0.85
            if macd_hist < 0 and macd_hist <= prev_macd_hist:
                sell_factors["momentum_expanding"] = 0.75
            if 32 <= rsi <= 55:
                sell_factors["rsi_healthy"] = 0.65
            if supertrend_direction == -1:
                sell_factors["supertrend_confirmed"] = 0.75
            if adx >= 25:
                sell_factors["strong_adx"] = 0.7

        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        min_factors = 4
        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.STRONG_BUY if buy_confidence >= 0.75 else Signal.BUY,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price - (2.0 * atr),
                take_profit=price + (3.5 * atr),
                metadata={"factors": buy_factors, "adx": adx, "volume_ratio": volume_ratio},
            )

        if len(sell_factors) >= min_factors and sell_confidence > buy_confidence:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.STRONG_SELL if sell_confidence >= 0.75 else Signal.SELL,
                confidence=sell_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price + (2.0 * atr),
                take_profit=price - (3.5 * atr),
                metadata={"factors": sell_factors, "adx": adx, "volume_ratio": volume_ratio},
            )

        return self._neutral_signal(df, symbol)

