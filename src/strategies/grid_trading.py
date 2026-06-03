"""
Grid Trading strategy — places buy/sell orders at regular price intervals.
Works best in ranging/sideways markets where price oscillates between
support and resistance. Profits from each grid level bounce.

How it works:
    1. Define a price range (upper/lower bounds) based on Bollinger Bands or Pivot Points
    2. Divide the range into N grid levels (default 10)
    3. BUY when price drops to a lower grid level
    4. SELL when price rises to an upper grid level
    5. Each grid level acts as a mini support/resistance zone

This strategy is inherently mean-reverting within the grid range.
It does NOT work well in strong trends (price breaks out of grid).
Regime filter disables grid trading in strong trends.
"""

import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading strategy for ranging markets.

    Uses Bollinger Bands to define the grid range and divides it into
    equal intervals. Generates BUY signals near lower grid levels and
    SELL signals near upper grid levels.
    """

    def __init__(self, config=None, grid_levels: int = 10):
        """
        Initialize grid trading strategy.

        Args:
            config: Strategy configuration
            grid_levels: Number of grid divisions between upper and lower bounds
        """
        super().__init__(config)
        self.grid_levels = grid_levels

    def get_name(self) -> str:
        return "grid_trading"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate grid trading signal based on price position within the grid.

        Grid range is defined by Bollinger Bands (upper/lower) or Pivot Points.
        BUY when price is in the lower 30% of the grid, SELL when in upper 30%.
        """
        if len(df) < 30:
            return self._neutral_signal(df, symbol)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]

        # --- Regime filter: only trade in ranging/low-volatility markets ---
        regime_str = curr.get("regime_str", "ranging")
        adx = float(curr.get("adx", 0))
        # Skip in strong trends — grid trading loses money when price trends out of range
        if adx > 30 or regime_str in ("strong_uptrend", "strong_downtrend"):
            return self._neutral_signal(df, symbol)

        # --- Define grid range using Bollinger Bands ---
        bb_upper = float(curr.get("bb_upper", price * 1.02))
        bb_lower = float(curr.get("bb_lower", price * 0.98))
        bb_pct = float(curr.get("bb_pct", 0.5))

        # Also use Pivot Points for grid boundaries if available
        pivot_r1 = float(curr.get("pivot_r1", bb_upper))
        pivot_s1 = float(curr.get("pivot_s1", bb_lower))

        # Use the tighter of BB and Pivot Points as grid boundaries
        grid_upper = min(bb_upper, pivot_r1)
        grid_lower = max(bb_lower, pivot_s1)
        grid_range = grid_upper - grid_lower

        if grid_range <= 0:
            return self._neutral_signal(df, symbol)

        # --- Calculate grid level position (0 = bottom, 1 = top) ---
        grid_position = (price - grid_lower) / grid_range
        grid_position = max(0.0, min(1.0, grid_position))

        # --- Grid level step size ---
        step = 1.0 / self.grid_levels
        # Which grid level are we at?
        current_level = int(grid_position / step)

        # --- ATR for stop/target ---
        atr = float(curr.get("atr", price * 0.02))

        # --- Additional confirmation indicators ---
        rsi = float(curr.get("rsi", 50))
        williams_r = float(curr.get("williams_r", -50))
        mfi = float(curr.get("mfi", 50))

        # --- Build confidence factors ---
        buy_factors = {}
        sell_factors = {}

        # BUY zone: price in lower 30% of grid
        if grid_position < 0.3:
            buy_factors["grid_lower_zone"] = 0.7
            # Stronger signal at lower grid levels
            if grid_position < 0.15:
                buy_factors["grid_bottom"] = 0.8

            # Confirmation from other indicators
            if rsi < 40:
                buy_factors["rsi_supports_buy"] = 0.5
            if williams_r < -70:
                buy_factors["williams_r_oversold"] = 0.5
            if mfi < 35:
                buy_factors["mfi_low"] = 0.5
            # Price bouncing off grid level (close > open after reaching lower zone)
            if curr["close"] > curr["open"]:
                buy_factors["bounce_candle"] = 0.4

        # SELL zone: price in upper 30% of grid
        elif grid_position > 0.7:
            sell_factors["grid_upper_zone"] = 0.7
            # Stronger signal at upper grid levels
            if grid_position > 0.85:
                sell_factors["grid_top"] = 0.8

            # Confirmation from other indicators
            if rsi > 60:
                sell_factors["rsi_supports_sell"] = 0.5
            if williams_r > -30:
                sell_factors["williams_r_overbought"] = 0.5
            if mfi > 65:
                sell_factors["mfi_high"] = 0.5
            # Price rejected from grid level (close < open after reaching upper zone)
            if curr["close"] < curr["open"]:
                sell_factors["rejection_candle"] = 0.4

        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        # Grid trading uses tighter stops (1x ATR) and closer targets (grid step)
        grid_step_price = grid_range / self.grid_levels

        min_factors = 2

        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            signal = Signal.STRONG_BUY if buy_confidence > 0.7 else Signal.BUY
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price - (1.0 * atr),
                take_profit=price + (2 * grid_step_price),
                metadata={
                    "grid_position": round(grid_position, 3),
                    "grid_level": current_level,
                    "grid_upper": round(grid_upper, 2),
                    "grid_lower": round(grid_lower, 2),
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
                stop_loss=price + (1.0 * atr),
                take_profit=price - (2 * grid_step_price),
                metadata={
                    "grid_position": round(grid_position, 3),
                    "grid_level": current_level,
                    "grid_upper": round(grid_upper, 2),
                    "grid_lower": round(grid_lower, 2),
                    "factors": sell_factors,
                },
            )

        return self._neutral_signal(df, symbol)
