"""
Momentum crossover strategy — designed to generate signals in ANY market condition.
Unlike trend-following or mean-reversion which require extreme conditions (high ADX,
extreme RSI, Bollinger Band touches), this strategy uses EMA crossovers and MACD
momentum changes that occur regularly in all market environments.

This strategy is the primary signal generator and ensures the system takes trades
even in neutral/ranging markets where other strategies remain silent.
"""

import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class MomentumCrossoverStrategy(BaseStrategy):
    """
    Momentum-based strategy using EMA crossovers and MACD momentum.

    Signals are generated when:
        BUY:
            - EMA-12 crosses above EMA-26 (or is above and widening)
            - MACD histogram is positive OR turning positive
            - Price is above its 20-period SMA (short-term bullish)

        SELL:
            - EMA-12 crosses below EMA-26 (or is below and widening)
            - MACD histogram is negative OR turning negative
            - Price is below its 20-period SMA (short-term bearish)

    This strategy intentionally has looser entry conditions than trend_following
    or mean_reversion to ensure signals are generated in normal market conditions.
    """

    def get_name(self) -> str:
        return "momentum_crossover"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate a momentum crossover signal from EMA/MACD data."""
        if len(df) < 30:
            return self._neutral_signal(df, symbol)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 3 else prev
        price = curr["close"]

        # --- EMA crossover signals ---
        ema_fast = curr.get("ema_12", price)
        ema_slow = curr.get("ema_26", price)
        prev_ema_fast = prev.get("ema_12", price)
        prev_ema_slow = prev.get("ema_26", price)

        # Handle potential Series objects from duplicate columns
        if hasattr(ema_fast, 'iloc'):
            ema_fast = float(ema_fast.iloc[0])
        if hasattr(ema_slow, 'iloc'):
            ema_slow = float(ema_slow.iloc[0])
        if hasattr(prev_ema_fast, 'iloc'):
            prev_ema_fast = float(prev_ema_fast.iloc[0])
        if hasattr(prev_ema_slow, 'iloc'):
            prev_ema_slow = float(prev_ema_slow.iloc[0])

        # Fresh crossover: EMA-12 just crossed above/below EMA-26
        bullish_cross = ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow
        bearish_cross = ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow

        # Widening gap: EMAs already crossed and the gap is growing (momentum increasing)
        ema_gap = ema_fast - ema_slow
        prev_ema_gap = prev_ema_fast - prev_ema_slow
        bullish_widening = ema_fast > ema_slow and ema_gap > prev_ema_gap
        bearish_widening = ema_fast < ema_slow and ema_gap < prev_ema_gap

        # --- MACD momentum ---
        macd_hist = float(curr.get("macd_histogram", 0))
        prev_macd_hist = float(prev.get("macd_histogram", 0))
        prev2_macd_hist = float(prev2.get("macd_histogram", 0))

        # Handle Series
        if hasattr(macd_hist, 'iloc'):
            macd_hist = float(macd_hist.iloc[0])
        if hasattr(prev_macd_hist, 'iloc'):
            prev_macd_hist = float(prev_macd_hist.iloc[0])
        if hasattr(prev2_macd_hist, 'iloc'):
            prev2_macd_hist = float(prev2_macd_hist.iloc[0])

        # MACD turning: histogram changing direction
        macd_turning_bullish = macd_hist > prev_macd_hist and prev_macd_hist <= prev2_macd_hist
        macd_turning_bearish = macd_hist < prev_macd_hist and prev_macd_hist >= prev2_macd_hist
        macd_positive = macd_hist > 0
        macd_negative = macd_hist < 0

        # --- Price vs SMA-20 (short-term trend direction) ---
        sma_20 = curr.get("sma_20", price)
        if hasattr(sma_20, 'iloc'):
            sma_20 = float(sma_20.iloc[0])
        price_above_sma = price > sma_20
        price_below_sma = price < sma_20

        # --- VWAP institutional bias ---
        vwap = curr.get("vwap", price)
        if hasattr(vwap, 'iloc'):
            vwap = float(vwap.iloc[0])
        price_above_vwap = price > vwap
        price_below_vwap = price < vwap

        # --- Supertrend direction ---
        st_dir = curr.get("supertrend_direction", 0)
        if hasattr(st_dir, 'iloc'):
            st_dir = float(st_dir.iloc[0])
        supertrend_bullish = st_dir == 1
        supertrend_bearish = st_dir == -1

        # --- RSI momentum (not extreme, just directional) ---
        rsi = float(curr.get("rsi", 50))
        prev_rsi = float(prev.get("rsi", 50))
        if hasattr(rsi, 'iloc'):
            rsi = float(rsi.iloc[0])
        if hasattr(prev_rsi, 'iloc'):
            prev_rsi = float(prev_rsi.iloc[0])

        rsi_rising = rsi > prev_rsi
        rsi_falling = rsi < prev_rsi
        # Don't buy into overbought, don't sell into oversold
        rsi_not_extreme_high = rsi < 72
        rsi_not_extreme_low = rsi > 28

        # --- Volume confirmation ---
        vol_ratio = float(curr.get("volume_ratio", 1.0))
        if hasattr(vol_ratio, 'iloc'):
            vol_ratio = float(vol_ratio.iloc[0])
        volume_ok = vol_ratio > 0.3  # Very loose volume filter

        # --- ATR for stop/target ---
        atr = float(curr.get("atr", price * 0.02))
        if hasattr(atr, 'iloc'):
            atr = float(atr.iloc[0])

        # --- Build confidence factors ---
        buy_factors = {}
        sell_factors = {}

        # BUY factors — any combination of 2+ triggers a trade
        if bullish_cross:
            buy_factors["ema_cross"] = 0.9
        elif bullish_widening:
            buy_factors["ema_widening"] = 0.6

        if macd_positive:
            buy_factors["macd_positive"] = 0.5
        if macd_turning_bullish:
            buy_factors["macd_turning"] = 0.7

        if price_above_sma and rsi_not_extreme_high:
            buy_factors["price_above_sma"] = 0.5

        if rsi_rising and rsi_not_extreme_high:
            buy_factors["rsi_rising"] = 0.4

        if volume_ok:
            buy_factors["volume_ok"] = 0.3

        # VWAP and Supertrend confirmation
        if price_above_vwap:
            buy_factors["vwap_bullish"] = 0.5
        if supertrend_bullish:
            buy_factors["supertrend_up"] = 0.6

        # SELL factors
        if bearish_cross:
            sell_factors["ema_cross"] = 0.9
        elif bearish_widening:
            sell_factors["ema_widening"] = 0.6

        if macd_negative:
            sell_factors["macd_negative"] = 0.5
        if macd_turning_bearish:
            sell_factors["macd_turning"] = 0.7

        if price_below_sma and rsi_not_extreme_low:
            sell_factors["price_below_sma"] = 0.5

        if rsi_falling and rsi_not_extreme_low:
            sell_factors["rsi_falling"] = 0.4

        if volume_ok:
            sell_factors["volume_ok"] = 0.3

        # VWAP and Supertrend confirmation
        if price_below_vwap:
            sell_factors["vwap_bearish"] = 0.5
        if supertrend_bearish:
            sell_factors["supertrend_down"] = 0.6

        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        # Require at least 2 confirming factors to reduce false signals and whipsaw trades
        min_factors = 2

        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            signal = Signal.STRONG_BUY if buy_confidence > 0.7 else Signal.BUY
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price - (1.5 * atr),
                take_profit=price + (2.5 * atr),
                metadata={
                    "ema_gap": round(ema_gap, 2),
                    "macd_hist": round(macd_hist, 4),
                    "rsi": round(rsi, 2),
                    "factors": buy_factors,
                },
            )
        elif len(sell_factors) >= min_factors and sell_confidence > buy_confidence:
            signal = Signal.STRONG_SELL if sell_confidence > 0.7 else Signal.SELL
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=sell_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price + (1.5 * atr),
                take_profit=price - (2.5 * atr),
                metadata={
                    "ema_gap": round(ema_gap, 2),
                    "macd_hist": round(macd_hist, 4),
                    "rsi": round(rsi, 2),
                    "factors": sell_factors,
                },
            )

        return self._neutral_signal(df, symbol)
