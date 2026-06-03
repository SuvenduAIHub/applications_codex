"""
Volatility strategy — trades based on volatility regime changes.
BTC is volatility-driven; this strategy detects when volatility is expanding
or contracting and trades the resulting directional moves.

Uses ATR, Historical Volatility, Bollinger Band Width, and BB+Keltner squeeze
to identify volatility breakouts and compression. Pairs with dynamic SL/TP
based on current volatility level.
"""

import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class VolatilityStrategy(BaseStrategy):
    """
    Volatility-based strategy for BTC and Gold.

    BUY when:
        - Volatility expanding after compression (BB width increasing)
        - Price breaking out in bullish direction during vol expansion
        - ATR expanding + Supertrend bullish
        - BB+Keltner squeeze just released upward

    SELL when:
        - Volatility expanding with bearish direction
        - Price breaking down during vol expansion
        - ATR expanding + Supertrend bearish
        - BB+Keltner squeeze just released downward

    Dynamic SL/TP scales with current volatility level.
    """

    def get_name(self) -> str:
        return "volatility"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate volatility-based signal from ATR, BB width, and squeeze indicators."""
        if len(df) < 30:
            return self._neutral_signal(df, symbol)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]

        # --- ATR expansion/contraction ---
        atr = float(curr.get("atr", price * 0.02))
        atr_prev = float(prev.get("atr", atr))
        norm_atr = float(curr.get("norm_atr_pct", 2.0))
        # ATR expanding = volatility increasing
        atr_expanding = atr > atr_prev * 1.03

        # --- Bollinger Band Width (volatility proxy) ---
        bb_width = float(curr.get("bb_width", 0))
        bb_width_prev = float(prev.get("bb_width", bb_width))
        # BB width increasing = volatility expanding
        bb_expanding = bb_width > bb_width_prev * 1.05

        # --- Historical Volatility ---
        hist_vol = float(curr.get("hist_volatility", 0))
        prev_hist_vol = float(prev.get("hist_volatility", 0))
        vol_increasing = hist_vol > prev_hist_vol

        # --- BB + Keltner Squeeze ---
        squeeze_on = bool(curr.get("squeeze_on", False))
        prev_squeeze = bool(prev.get("squeeze_on", False))
        squeeze_released = prev_squeeze and not squeeze_on

        # --- Direction indicators (which way is the vol expansion pushing price?) ---
        # Supertrend direction
        st_dir = float(curr.get("supertrend_direction", 0))
        supertrend_bullish = st_dir == 1
        supertrend_bearish = st_dir == -1

        # MACD direction for momentum
        macd_hist = float(curr.get("macd_histogram", 0))
        macd_bullish = macd_hist > 0
        macd_bearish = macd_hist < 0

        # Price vs BB middle (above = bullish expansion, below = bearish)
        bb_middle = float(curr.get("bb_middle", price))
        price_above_middle = price > bb_middle
        price_below_middle = price < bb_middle

        # VWAP direction
        vwap = float(curr.get("vwap", price))
        price_above_vwap = price > vwap
        price_below_vwap = price < vwap

        # --- Build confidence factors ---
        buy_factors = {}
        sell_factors = {}

        # Volatility expansion signals (direction-agnostic)
        vol_expanding = atr_expanding or bb_expanding or vol_increasing

        if vol_expanding and price_above_middle:
            if atr_expanding:
                buy_factors["atr_expansion"] = 0.7
            if bb_expanding:
                buy_factors["bb_expanding"] = 0.6
            if vol_increasing:
                buy_factors["hist_vol_up"] = 0.5

        if vol_expanding and price_below_middle:
            if atr_expanding:
                sell_factors["atr_expansion"] = 0.7
            if bb_expanding:
                sell_factors["bb_expanding"] = 0.6
            if vol_increasing:
                sell_factors["hist_vol_up"] = 0.5

        # Squeeze release — strongest signal
        if squeeze_released:
            if price_above_middle:
                buy_factors["squeeze_release"] = 0.9
            else:
                sell_factors["squeeze_release"] = 0.9

        # Direction confirmation
        if supertrend_bullish:
            buy_factors["supertrend_up"] = 0.6
        elif supertrend_bearish:
            sell_factors["supertrend_down"] = 0.6

        if macd_bullish:
            buy_factors["macd_positive"] = 0.5
        elif macd_bearish:
            sell_factors["macd_negative"] = 0.5

        if price_above_vwap:
            buy_factors["vwap_bullish"] = 0.5
        elif price_below_vwap:
            sell_factors["vwap_bearish"] = 0.5

        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        # Dynamic SL/TP based on current volatility (wider stops in high vol)
        atr_multiplier = max(1.5, min(3.0, norm_atr))  # Scale 1.5x-3x ATR

        min_factors = 2

        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            signal = Signal.STRONG_BUY if buy_confidence > 0.7 else Signal.BUY
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price - (atr_multiplier * atr),
                take_profit=price + (atr_multiplier * 1.5 * atr),
                metadata={
                    "atr": round(atr, 2),
                    "bb_width": round(bb_width, 4),
                    "squeeze_released": squeeze_released,
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
                stop_loss=price + (atr_multiplier * atr),
                take_profit=price - (atr_multiplier * 1.5 * atr),
                metadata={
                    "atr": round(atr, 2),
                    "bb_width": round(bb_width, 4),
                    "squeeze_released": squeeze_released,
                    "factors": sell_factors,
                },
            )

        return self._neutral_signal(df, symbol)
