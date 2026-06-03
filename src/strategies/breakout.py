"""
Breakout strategy implementation.
Identifies price breakouts from consolidation zones using
support/resistance levels, volume confirmation, ATR expansion,
BB+Keltner squeeze detection, Donchian Channels, and VWAP.
Very powerful for Bitcoin which frequently consolidates before explosive moves.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy that detects price breakouts from consolidation ranges.

    Entry conditions (BUY - bullish breakout):
        - Price breaks above recent high (resistance level)
        - Volume surge confirms the breakout (volume > 1.5x average)
        - ATR expanding (increasing volatility confirms breakout momentum)
        - Bollinger Band width was contracting (squeeze before breakout)

    Entry conditions (SELL - bearish breakdown):
        - Price breaks below recent low (support level)
        - Volume surge confirms the breakdown
        - ATR expanding

    Risk management:
        - Stop-loss placed at the breakout level (previous support/resistance)
        - Take-profit at 2x the consolidation range height
    """

    def __init__(self, config=None, lookback: int = 20):
        """
        Initialize the breakout strategy.

        Args:
            config: Strategy configuration
            lookback: Number of candles to look back for range detection
        """
        super().__init__(config)
        self.lookback = lookback

    def get_name(self) -> str:
        return "breakout"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate a breakout signal.

        Args:
            df: DataFrame with indicators (must include atr, volume_ratio, bb_width)
            symbol: Trading pair symbol

        Returns:
            TradeSignal with breakout direction and confidence
        """
        if len(df) < self.lookback + 2:
            return self._neutral_signal(df, symbol)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]

        # --- Determine consolidation range ---
        lookback_data = df.iloc[-(self.lookback + 1):-1]  # Exclude current candle
        range_high = lookback_data["high"].max()
        range_low = lookback_data["low"].min()
        range_height = range_high - range_low

        # Range as percentage of price (narrow range = consolidation)
        range_pct = (range_height / price) * 100

        # --- Breakout detection ---
        bullish_breakout = price > range_high
        bearish_breakdown = price < range_low

        # Check previous candle was near the range (confirms relatively fresh breakout) — 2% tolerance added
        tolerance = range_height * 0.02
        prev_in_range = prev["close"] <= (range_high + tolerance) and prev["close"] >= (range_low - tolerance)

        # --- Volume confirmation ---
        vol_ratio = curr.get("volume_ratio", 1.0)
        volume_spike = vol_ratio > self.config.volume_spike_multiplier

        # --- ATR expansion (volatility increasing = momentum behind the move) ---
        atr = curr.get("atr", 0)
        atr_prev = prev.get("atr", 0)
        atr_expanding = atr > atr_prev * 1.05  # ATR grew by at least 5%

        # --- Bollinger Band squeeze detection ---
        # A squeeze (narrow bands) followed by expansion often precedes breakouts
        bb_width = curr.get("bb_width", 0)
        bb_width_sma = df["bb_width"].rolling(window=self.lookback).mean().iloc[-1] if "bb_width" in df else 0
        was_squeezed = bb_width < bb_width_sma * 0.8  # Width below 80% of average

        # --- Candle body strength ---
        body = abs(curr["close"] - curr["open"])
        wick = curr["high"] - curr["low"]
        body_ratio = body / wick if wick > 0 else 0
        strong_candle = body_ratio > 0.6  # Strong body (minimal wicks)

        # --- Build confidence factors ---
        buy_factors = {}
        sell_factors = {}

        if bullish_breakout and prev_in_range:
            buy_factors["breakout"] = 0.9
            if volume_spike:
                buy_factors["volume_confirmation"] = 0.85
            if atr_expanding:
                buy_factors["atr_expansion"] = 0.7
            if was_squeezed:
                buy_factors["bb_squeeze"] = 0.6
            if strong_candle:
                buy_factors["candle_strength"] = 0.5

        if bearish_breakdown and prev_in_range:
            sell_factors["breakdown"] = 0.9
            if volume_spike:
                sell_factors["volume_confirmation"] = 0.85
            if atr_expanding:
                sell_factors["atr_expansion"] = 0.7
            if was_squeezed:
                sell_factors["bb_squeeze"] = 0.6
            if strong_candle:
                sell_factors["candle_strength"] = 0.5

        # --- BB + Keltner Squeeze release detection (professional breakout signal) ---
        squeeze_on = curr.get("squeeze_on", False)
        prev_squeeze = prev.get("squeeze_on", False)
        # Squeeze just released — Bollinger Bands expanding outside Keltner = breakout imminent
        if prev_squeeze and not squeeze_on:
            if price > curr.get("bb_middle", price):
                buy_factors["squeeze_release_up"] = 0.8
            else:
                sell_factors["squeeze_release_down"] = 0.8

        # --- Donchian Channel breakout (classic trend breakout signal) ---
        donchian_upper = curr.get("donchian_upper", range_high)
        donchian_lower = curr.get("donchian_lower", range_low)
        if hasattr(donchian_upper, 'iloc'):
            donchian_upper = float(donchian_upper.iloc[0])
        if hasattr(donchian_lower, 'iloc'):
            donchian_lower = float(donchian_lower.iloc[0])
        if price >= donchian_upper:
            buy_factors["donchian_breakout"] = 0.75
        elif price <= donchian_lower:
            sell_factors["donchian_breakdown"] = 0.75

        # --- VWAP confirmation (breakout above VWAP = institutional support) ---
        vwap = curr.get("vwap", price)
        if hasattr(vwap, 'iloc'):
            vwap = float(vwap.iloc[0])
        if price > vwap and bullish_breakout:
            buy_factors["vwap_above"] = 0.6
        elif price < vwap and bearish_breakdown:
            sell_factors["vwap_below"] = 0.6

        # --- ROC momentum (Rate of Change confirms momentum behind breakout) ---
        roc = curr.get("roc", 0)
        if hasattr(roc, 'iloc'):
            roc = float(roc.iloc[0])
        if roc > 1.0:
            buy_factors["roc_positive"] = 0.5
        elif roc < -1.0:
            sell_factors["roc_negative"] = 0.5

        # --- CCI breakout confirmation (strong momentum beyond +100/-100) ---
        cci = float(curr.get("cci", 0))
        if cci > 100:
            buy_factors["cci_strong_up"] = 0.6
        elif cci < -100:
            sell_factors["cci_strong_down"] = 0.6

        # --- Pivot Point breakout (price breaking above R1 or below S1) ---
        pivot_r1 = float(curr.get("pivot_r1", range_high))
        pivot_s1 = float(curr.get("pivot_s1", range_low))
        if price > pivot_r1:
            buy_factors["pivot_r1_break"] = 0.6
        elif price < pivot_s1:
            sell_factors["pivot_s1_break"] = 0.6

        # --- MFI confirms volume pressure behind breakout ---
        mfi = float(curr.get("mfi", 50))
        if mfi > 60 and bullish_breakout:
            buy_factors["mfi_buying_pressure"] = 0.5
        elif mfi < 40 and bearish_breakdown:
            sell_factors["mfi_selling_pressure"] = 0.5

        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        min_factors = self.config.min_signal_confirmations

        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            signal = Signal.STRONG_BUY if buy_confidence > 0.75 else Signal.BUY
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                # Stop-loss at the top of the range (now support)
                stop_loss=range_high - (0.5 * range_height),
                # Take-profit at 2x the range height from breakout
                take_profit=range_high + (2 * range_height),
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": range_pct,
                    "volume_ratio": vol_ratio,
                    "was_squeezed": was_squeezed,
                    "factors": buy_factors,
                },
            )
        elif len(sell_factors) >= min_factors and sell_confidence > buy_confidence:
            signal = Signal.STRONG_SELL if sell_confidence > 0.75 else Signal.SELL
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=sell_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=range_low + (0.5 * range_height),
                take_profit=range_low - (2 * range_height),
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": range_pct,
                    "volume_ratio": vol_ratio,
                    "was_squeezed": was_squeezed,
                    "factors": sell_factors,
                },
            )

        return self._neutral_signal(df, symbol)

    def _neutral_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Return a HOLD signal when no breakout is detected."""
        price = df.iloc[-1]["close"] if len(df) > 0 else 0.0
        return TradeSignal(
            symbol=symbol,
            signal=Signal.HOLD,
            confidence=0.0,
            strategy_name=self.get_name(),
            price=price,
        )
