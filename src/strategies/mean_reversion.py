"""
Mean-reversion strategy implementation.
Uses Bollinger Bands, RSI extremes, Stochastic RSI, VWAP deviation,
and price deviation from moving averages to identify overbought/oversold
conditions for counter-trend entries. Enhanced with professional-grade
indicators for Gold (which mean-reverts frequently).
"""

import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-reversion strategy using Bollinger Bands and RSI extremes.

    Entry conditions (BUY - oversold bounce):
        - Price touches or crosses below lower Bollinger Band
        - RSI is below oversold threshold (30)
        - Price shows reversal candle (close > open after touching lower band)
        - Volume is not excessively low (validates the move)

    Entry conditions (SELL - overbought reversal):
        - Price touches or crosses above upper Bollinger Band
        - RSI is above overbought threshold (70)
        - Price shows reversal candle (close < open after touching upper band)

    Works best in ranging/sideways markets; filtered out during strong trends.
    """

    def get_name(self) -> str:
        return "mean_reversion"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate a mean-reversion signal.

        Args:
            df: DataFrame with indicators (must include bb_*, rsi, atr)
            symbol: Trading pair symbol

        Returns:
            TradeSignal with mean-reversion direction and confidence
        """
        if len(df) < 3:
            return self._neutral_signal(df, symbol)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]

        # --- Bollinger Band conditions ---
        bb_lower = curr.get("bb_lower", price * 0.98)
        bb_upper = curr.get("bb_upper", price * 1.02)
        bb_middle = curr.get("bb_middle", price)
        bb_pct = curr.get("bb_pct", 0.5)

        # Price at or below lower band — relaxed threshold for more signals
        at_lower_band = price <= bb_lower or bb_pct <= 0.15
        # Price at or above upper band — relaxed threshold for more signals
        at_upper_band = price >= bb_upper or bb_pct >= 0.85

        # Previous candle was also near the band (confirming the test) — widened zone
        prev_near_lower = prev.get("bb_pct", 0.5) <= 0.25
        prev_near_upper = prev.get("bb_pct", 0.5) >= 0.75

        # --- RSI conditions ---
        rsi = curr.get("rsi", 50)
        rsi_oversold = rsi < self.config.rsi_oversold
        rsi_overbought = rsi > self.config.rsi_overbought

        # RSI divergence: price makes new low but RSI doesn't (bullish divergence) — widened RSI zone
        rsi_prev = prev.get("rsi", 50)
        price_lower = price < prev["close"]
        rsi_higher = rsi > rsi_prev
        bullish_divergence = price_lower and rsi_higher and rsi < 45

        price_higher = price > prev["close"]
        rsi_lower = rsi < rsi_prev
        bearish_divergence = price_higher and rsi_lower and rsi > 55

        # --- Reversal candle detection ---
        bullish_reversal = curr["close"] > curr["open"] and prev["close"] < prev["open"]
        bearish_reversal = curr["close"] < curr["open"] and prev["close"] > prev["open"]

        # --- Volume confirmation ---
        vol_ratio = curr.get("volume_ratio", 1.0)
        volume_adequate = vol_ratio > 0.5  # Not abnormally low volume

        # --- Regime filter: disabled for M5 scalping to maximize trade frequency ---
        # On 5-minute candles, "strong trends" are very short-lived and
        # mean-reversion entries still work with tight stop-losses
        # (Previously skipped strong_downtrend; now we trade through all regimes)

        # --- ATR for stop/target ---
        atr = curr.get("atr", price * 0.02)

        # --- MACD momentum confirmation ---
        macd_hist = curr.get("macd_histogram", 0)
        macd_hist_prev = prev.get("macd_histogram", 0)
        macd_turning_up = macd_hist > macd_hist_prev and macd_hist < 0  # Histogram turning up from negative
        macd_turning_down = macd_hist < macd_hist_prev and macd_hist > 0  # Histogram turning down from positive

        # --- Price below/above middle Bollinger Band (mean reversion target) ---
        below_middle_band = price < bb_middle
        above_middle_band = price > bb_middle

        # --- Build confidence factors ---
        buy_factors = {}
        sell_factors = {}

        # Oversold bounce conditions — widened criteria for more trade opportunities
        if at_lower_band:
            buy_factors["bb_lower_touch"] = 0.8
        if rsi_oversold:
            buy_factors["rsi_oversold"] = 0.8
        if bullish_reversal:
            buy_factors["reversal_candle"] = 0.7
        if bullish_divergence:
            buy_factors["rsi_divergence"] = 0.9
        if prev_near_lower:
            buy_factors["band_confirmation"] = 0.5
        if volume_adequate:
            buy_factors["volume_ok"] = 0.4
        # MACD turning up from negative — momentum shifting bullish
        if macd_turning_up and below_middle_band:
            buy_factors["macd_turning_up"] = 0.6

        # Overbought reversal conditions
        if at_upper_band:
            sell_factors["bb_upper_touch"] = 0.8
        if rsi_overbought:
            sell_factors["rsi_overbought"] = 0.8
        if bearish_reversal:
            sell_factors["reversal_candle"] = 0.7
        if bearish_divergence:
            sell_factors["rsi_divergence"] = 0.9
        if prev_near_upper:
            sell_factors["band_confirmation"] = 0.5
        if volume_adequate:
            sell_factors["volume_ok"] = 0.4
        # MACD turning down from positive — momentum shifting bearish
        if macd_turning_down and above_middle_band:
            sell_factors["macd_turning_down"] = 0.6

        # --- Stochastic RSI (faster reversal detection than plain RSI) ---
        stoch_rsi_k = curr.get("stoch_rsi_k", 50)
        stoch_rsi_d = curr.get("stoch_rsi_d", 50)
        if hasattr(stoch_rsi_k, 'iloc'):
            stoch_rsi_k = float(stoch_rsi_k.iloc[0])
        if hasattr(stoch_rsi_d, 'iloc'):
            stoch_rsi_d = float(stoch_rsi_d.iloc[0])
        # Stochastic RSI oversold with K crossing above D (bullish reversal)
        prev_stoch_k = prev.get("stoch_rsi_k", 50)
        if hasattr(prev_stoch_k, 'iloc'):
            prev_stoch_k = float(prev_stoch_k.iloc[0])
        if stoch_rsi_k < 20 or (stoch_rsi_k > stoch_rsi_d and prev_stoch_k <= stoch_rsi_d and stoch_rsi_k < 40):
            buy_factors["stoch_rsi_oversold"] = 0.7
        if stoch_rsi_k > 80 or (stoch_rsi_k < stoch_rsi_d and prev_stoch_k >= stoch_rsi_d and stoch_rsi_k > 60):
            sell_factors["stoch_rsi_overbought"] = 0.7

        # --- VWAP deviation (price far from institutional average = reversion likely) ---
        vwap = curr.get("vwap", price)
        if hasattr(vwap, 'iloc'):
            vwap = float(vwap.iloc[0])
        vwap_dev = (price - vwap) / vwap if vwap > 0 else 0
        # Price significantly below VWAP → oversold relative to institutions
        if vwap_dev < -0.005:
            buy_factors["vwap_below"] = 0.6
        # Price significantly above VWAP → overbought relative to institutions
        elif vwap_dev > 0.005:
            sell_factors["vwap_above"] = 0.6

        # --- OBV divergence (smart money confirmation) ---
        obv = curr.get("obv", 0)
        prev_obv = prev.get("obv", 0)
        if hasattr(obv, 'iloc'):
            obv = float(obv.iloc[0])
        if hasattr(prev_obv, 'iloc'):
            prev_obv = float(prev_obv.iloc[0])
        # Price falling but OBV rising = smart money accumulating (bullish for mean reversion buy)
        if price < prev["close"] and obv > prev_obv:
            buy_factors["obv_accumulation"] = 0.6
        # Price rising but OBV falling = smart money distributing (bearish)
        elif price > prev["close"] and obv < prev_obv:
            sell_factors["obv_distribution"] = 0.6

        # --- Williams %R (oversold/overbought momentum) ---
        williams_r = float(curr.get("williams_r", -50))
        if williams_r < -80:
            buy_factors["williams_r_oversold"] = 0.7
        elif williams_r > -20:
            sell_factors["williams_r_overbought"] = 0.7

        # --- CCI (Commodity Channel Index — especially good for Gold) ---
        cci = float(curr.get("cci", 0))
        if cci < -100:
            buy_factors["cci_oversold"] = 0.6
        elif cci > 100:
            sell_factors["cci_overbought"] = 0.6

        # --- MFI (volume-weighted RSI — confirms buying/selling pressure) ---
        mfi = float(curr.get("mfi", 50))
        if mfi < 20:
            buy_factors["mfi_oversold"] = 0.7
        elif mfi > 80:
            sell_factors["mfi_overbought"] = 0.7

        # --- Pivot Points (price near support = buy, near resistance = sell) ---
        pivot_s1 = float(curr.get("pivot_s1", price * 0.99))
        pivot_r1 = float(curr.get("pivot_r1", price * 1.01))
        if price <= pivot_s1 * 1.002:
            buy_factors["near_pivot_support"] = 0.5
        elif price >= pivot_r1 * 0.998:
            sell_factors["near_pivot_resistance"] = 0.5

        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        min_factors = self.config.min_signal_confirmations

        # Mean reversion targets the middle Bollinger Band
        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            signal = Signal.STRONG_BUY if buy_confidence > 0.75 else Signal.BUY
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price - (1.5 * atr),    # Tighter stop for mean reversion
                take_profit=bb_middle,              # Target: middle band
                metadata={
                    "bb_pct": bb_pct,
                    "rsi": rsi,
                    "bullish_divergence": bullish_divergence,
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
                stop_loss=price + (1.5 * atr),
                take_profit=bb_middle,              # Target: middle band
                metadata={
                    "bb_pct": bb_pct,
                    "rsi": rsi,
                    "bearish_divergence": bearish_divergence,
                    "factors": sell_factors,
                },
            )

        return self._neutral_signal(df, symbol)

    def _neutral_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Return a HOLD signal when no mean-reversion opportunity is found."""
        price = df.iloc[-1]["close"] if len(df) > 0 else 0.0
        return TradeSignal(
            symbol=symbol,
            signal=Signal.HOLD,
            confidence=0.0,
            strategy_name=self.get_name(),
            price=price,
        )
