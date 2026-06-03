"""
Trend-following strategy implementation.
Uses EMA crossovers, MACD confirmation, ADX for trend strength,
plus Supertrend, Ichimoku Cloud, Parabolic SAR, and VWAP for
institutional bias filtering. Enhanced with professional-grade
indicators for stronger signal confirmation.
"""

import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-following strategy based on EMA crossovers with MACD confirmation.

    Entry conditions (BUY):
        - Fast EMA crosses above slow EMA
        - MACD histogram is positive and rising
        - RSI is above 40 (not oversold into a downtrend)
        - Price is above SMA 50 (medium-term trend is up)

    Entry conditions (SELL):
        - Fast EMA crosses below slow EMA
        - MACD histogram is negative and falling
        - RSI is below 60 (not overbought into an uptrend)
        - Price is below SMA 50

    Stop-loss: Set at 2x ATR below entry (buy) or above entry (sell)
    Take-profit: Set at 3x ATR from entry
    """

    def get_name(self) -> str:
        return "trend_following"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate a trend-following signal from the indicator-enriched DataFrame.

        Args:
            df: DataFrame with indicators (must include ema_fast, ema_slow, macd, rsi, atr)
            symbol: Trading pair symbol

        Returns:
            TradeSignal with trend-based direction and confidence
        """
        if len(df) < 3:
            return self._neutral_signal(df, symbol)

        # Get the latest and previous values
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]

        # --- Check EMA crossover ---
        ema_cross_up = (prev["ema_fast"] <= prev["ema_slow"]) and (curr["ema_fast"] > curr["ema_slow"])
        ema_cross_down = (prev["ema_fast"] >= prev["ema_slow"]) and (curr["ema_fast"] < curr["ema_slow"])
        ema_bullish = curr["ema_fast"] > curr["ema_slow"]
        ema_bearish = curr["ema_fast"] < curr["ema_slow"]

        # --- Check MACD confirmation ---
        macd_bullish = curr.get("macd_histogram", 0) > 0
        macd_rising = curr.get("macd_histogram", 0) > prev.get("macd_histogram", 0)
        macd_bearish = curr.get("macd_histogram", 0) < 0
        macd_falling = curr.get("macd_histogram", 0) < prev.get("macd_histogram", 0)

        # --- Check RSI filter ---
        rsi = curr.get("rsi", 50)
        rsi_supports_buy = rsi > 40 and rsi < 75
        rsi_supports_sell = rsi < 60 and rsi > 25

        # --- Check price vs SMA 50 ---
        sma50 = curr.get("sma_50", price)
        above_sma50 = price > sma50
        below_sma50 = price < sma50

        # --- ADX trend strength filter (relaxed for M5 scalping) ---
        adx = curr.get("adx", 0)
        # Handle case where adx might be a Series (duplicate columns)
        adx_val = float(adx) if not isinstance(adx, (pd.Series,)) else float(adx.iloc[0]) if len(adx) > 0 else 0
        strong_trend = adx_val > 12  # ADX > 12 — lowered from 20 to catch more M5 trends
        very_strong_trend = adx_val > 25  # ADX > 25 is strong on M5

        # Skip only if ADX is extremely weak (no discernible direction at all)
        if not strong_trend:
            return self._neutral_signal(df, symbol)

        # --- ATR for stop-loss and take-profit ---
        atr = curr.get("atr", price * 0.02)

        # --- Build confidence factors ---
        buy_factors = {}
        sell_factors = {}

        # EMA crossover is the primary signal
        if ema_cross_up:
            buy_factors["ema_crossover"] = 0.9
        elif ema_bullish:
            buy_factors["ema_alignment"] = 0.6

        if ema_cross_down:
            sell_factors["ema_crossover"] = 0.9
        elif ema_bearish:
            sell_factors["ema_alignment"] = 0.6

        # MACD confirmation
        if macd_bullish and macd_rising:
            buy_factors["macd"] = 0.8
        elif macd_bullish:
            buy_factors["macd"] = 0.5

        if macd_bearish and macd_falling:
            sell_factors["macd"] = 0.8
        elif macd_bearish:
            sell_factors["macd"] = 0.5

        # RSI filter
        if rsi_supports_buy:
            buy_factors["rsi_filter"] = 0.6
        if rsi_supports_sell:
            sell_factors["rsi_filter"] = 0.6

        # Price vs SMA 50
        if above_sma50:
            buy_factors["trend_alignment"] = 0.7
        if below_sma50:
            sell_factors["trend_alignment"] = 0.7

        # ADX trend strength adds confidence in strong trends
        if very_strong_trend:
            buy_factors["adx_strong"] = 0.8
            sell_factors["adx_strong"] = 0.8
        elif strong_trend:
            buy_factors["adx_adequate"] = 0.6
            sell_factors["adx_adequate"] = 0.6

        # --- VWAP institutional bias filter ---
        vwap = curr.get("vwap", price)
        if hasattr(vwap, 'iloc'):
            vwap = float(vwap.iloc[0])
        if price > vwap:
            buy_factors["vwap_bullish"] = 0.6
        elif price < vwap:
            sell_factors["vwap_bearish"] = 0.6

        # --- Supertrend confirmation ---
        st_dir = curr.get("supertrend_direction", 0)
        if hasattr(st_dir, 'iloc'):
            st_dir = float(st_dir.iloc[0])
        if st_dir == 1:
            buy_factors["supertrend_bullish"] = 0.7
        elif st_dir == -1:
            sell_factors["supertrend_bearish"] = 0.7

        # --- Ichimoku Cloud confirmation (price above/below cloud) ---
        senkou_a = curr.get("ichimoku_senkou_a", price)
        senkou_b = curr.get("ichimoku_senkou_b", price)
        if hasattr(senkou_a, 'iloc'):
            senkou_a = float(senkou_a.iloc[0])
        if hasattr(senkou_b, 'iloc'):
            senkou_b = float(senkou_b.iloc[0])
        cloud_top = max(senkou_a, senkou_b) if not (pd.isna(senkou_a) or pd.isna(senkou_b)) else price
        cloud_bottom = min(senkou_a, senkou_b) if not (pd.isna(senkou_a) or pd.isna(senkou_b)) else price
        if price > cloud_top:
            buy_factors["ichimoku_above_cloud"] = 0.7
        elif price < cloud_bottom:
            sell_factors["ichimoku_below_cloud"] = 0.7

        # --- Parabolic SAR direction ---
        psar_dir = curr.get("psar_direction", 0)
        if hasattr(psar_dir, 'iloc'):
            psar_dir = float(psar_dir.iloc[0])
        if psar_dir == 1:
            buy_factors["psar_bullish"] = 0.5
        elif psar_dir == -1:
            sell_factors["psar_bearish"] = 0.5

        # Determine signal direction
        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        # Require minimum number of confirming factors
        min_factors = self.config.min_signal_confirmations

        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            signal = Signal.STRONG_BUY if buy_confidence > 0.75 else Signal.BUY
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price - (2 * atr),
                take_profit=price + (4 * atr),  # 4:2 reward:risk ratio
                metadata={
                    "ema_cross_up": ema_cross_up,
                    "macd_bullish": macd_bullish,
                    "rsi": rsi,
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
                stop_loss=price + (2 * atr),
                take_profit=price - (4 * atr),  # 4:2 reward:risk ratio
                metadata={
                    "ema_cross_down": ema_cross_down,
                    "macd_bearish": macd_bearish,
                    "rsi": rsi,
                    "factors": sell_factors,
                },
            )

        return self._neutral_signal(df, symbol)

    def _neutral_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Return a HOLD signal when no clear trend is detected."""
        price = df.iloc[-1]["close"] if len(df) > 0 else 0.0
        return TradeSignal(
            symbol=symbol,
            signal=Signal.HOLD,
            confidence=0.0,
            strategy_name=self.get_name(),
            price=price,
        )
