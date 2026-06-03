"""
Market regime detection module.
Identifies whether the market is in a trend or range-bound state,
and detects risk-on vs risk-off macro environments using BTC-Gold correlation.
"""

from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    """Identified market regime states."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"


class MacroRegime(Enum):
    """Macro regime based on BTC-Gold correlation analysis."""
    RISK_ON = "risk_on"      # Both BTC and Gold rising, or BTC outperforming
    RISK_OFF = "risk_off"    # Gold outperforming BTC, flight to safety
    NEUTRAL = "neutral"      # No clear macro signal
    DIVERGENT = "divergent"  # BTC and Gold moving in opposite directions


def detect_trend_regime(
    close: pd.Series,
    sma_short: int = 20,
    sma_long: int = 50,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    Detect the current market trend regime using moving average alignment
    and ADX (Average Directional Index) for trend strength.

    Args:
        close: Close price series
        sma_short: Short SMA period for trend direction
        sma_long: Long SMA period for trend direction
        adx_period: ADX calculation period
        adx_threshold: ADX threshold to distinguish trending vs ranging

    Returns:
        Series of MarketRegime enum values
    """
    # Calculate SMAs for trend direction
    sma_s = close.rolling(window=sma_short).mean()
    sma_l = close.rolling(window=sma_long).mean()

    # Calculate ADX for trend strength
    adx = _compute_adx(close, adx_period)

    # Price position relative to moving averages
    price_above_short = close > sma_s
    price_above_long = close > sma_l
    short_above_long = sma_s > sma_l

    # Volatility measure for high-vol detection
    returns = close.pct_change()
    rolling_vol = returns.rolling(window=adx_period).std()
    vol_threshold = rolling_vol.rolling(window=50).mean() + 2 * rolling_vol.rolling(window=50).std()

    regimes = []
    for i in range(len(close)):
        if i < sma_long:
            regimes.append(MarketRegime.RANGING)
            continue

        is_high_vol = rolling_vol.iloc[i] > vol_threshold.iloc[i] if pd.notna(vol_threshold.iloc[i]) else False

        if is_high_vol:
            regimes.append(MarketRegime.HIGH_VOLATILITY)
        elif adx.iloc[i] < adx_threshold:
            # Low ADX = range-bound market
            regimes.append(MarketRegime.RANGING)
        elif price_above_short.iloc[i] and price_above_long.iloc[i] and short_above_long.iloc[i]:
            # All aligned bullish with strong trend
            if adx.iloc[i] > 40:
                regimes.append(MarketRegime.STRONG_UPTREND)
            else:
                regimes.append(MarketRegime.UPTREND)
        elif not price_above_short.iloc[i] and not price_above_long.iloc[i] and not short_above_long.iloc[i]:
            # All aligned bearish with strong trend
            if adx.iloc[i] > 40:
                regimes.append(MarketRegime.STRONG_DOWNTREND)
            else:
                regimes.append(MarketRegime.DOWNTREND)
        else:
            regimes.append(MarketRegime.RANGING)

    return pd.Series(regimes, index=close.index, name="regime")


def detect_macro_regime(
    btc_close: pd.Series,
    gold_close: pd.Series,
    correlation_window: int = 30,
    return_window: int = 14,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect macro risk-on vs risk-off regime by analyzing BTC-Gold dynamics.

    Risk-on: BTC outperforming Gold, positive correlation rising
    Risk-off: Gold outperforming BTC, flight to safety
    Divergent: BTC and Gold moving in opposite directions

    Args:
        btc_close: BTC/USDT close price series
        gold_close: XAU/USD close price series
        correlation_window: Rolling window for correlation calculation
        return_window: Window for comparing relative performance

    Returns:
        Tuple of (macro_regime Series, rolling_correlation Series)
    """
    # Calculate rolling returns for both assets
    btc_returns = btc_close.pct_change(return_window)
    gold_returns = gold_close.pct_change(return_window)

    # Rolling correlation between BTC and Gold returns
    rolling_corr = btc_close.pct_change().rolling(window=correlation_window).corr(
        gold_close.pct_change()
    )

    # Relative performance: BTC return - Gold return
    relative_perf = btc_returns - gold_returns

    regimes = []
    for i in range(len(btc_close)):
        if i < max(correlation_window, return_window):
            regimes.append(MacroRegime.NEUTRAL)
            continue

        corr = rolling_corr.iloc[i] if pd.notna(rolling_corr.iloc[i]) else 0
        btc_ret = btc_returns.iloc[i] if pd.notna(btc_returns.iloc[i]) else 0
        gold_ret = gold_returns.iloc[i] if pd.notna(gold_returns.iloc[i]) else 0

        # Determine macro regime based on correlation and relative performance
        if corr > 0.3 and btc_ret > 0 and gold_ret > 0:
            # Both assets rising together - risk-on environment
            regimes.append(MacroRegime.RISK_ON)
        elif gold_ret > btc_ret and gold_ret > 0:
            # Gold outperforming - risk-off / flight to safety
            regimes.append(MacroRegime.RISK_OFF)
        elif corr < -0.3:
            # Negative correlation - assets diverging
            regimes.append(MacroRegime.DIVERGENT)
        elif btc_ret > gold_ret and btc_ret > 0:
            # BTC outperforming - risk-on
            regimes.append(MacroRegime.RISK_ON)
        else:
            regimes.append(MacroRegime.NEUTRAL)

    macro_series = pd.Series(regimes, index=btc_close.index, name="macro_regime")
    return macro_series, rolling_corr


def compute_regime_features(
    df: pd.DataFrame,
    btc_close: pd.Series = None,
    gold_close: pd.Series = None,
) -> pd.DataFrame:
    """
    Add regime detection features to an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame to enhance
        btc_close: BTC close prices (for cross-asset correlation; optional)
        gold_close: Gold close prices (for cross-asset correlation; optional)

    Returns:
        DataFrame with regime columns added
    """
    result = df.copy()

    # Detect trend regime for this asset
    result["regime"] = detect_trend_regime(df["close"])
    result["regime_str"] = result["regime"].apply(lambda r: r.value)

    # Encode regime as numeric for ML models
    regime_map = {
        MarketRegime.STRONG_DOWNTREND: -2,
        MarketRegime.DOWNTREND: -1,
        MarketRegime.RANGING: 0,
        MarketRegime.UPTREND: 1,
        MarketRegime.STRONG_UPTREND: 2,
        MarketRegime.HIGH_VOLATILITY: 0,
    }
    result["regime_numeric"] = result["regime"].map(regime_map)

    # Add macro regime if both BTC and Gold data are available
    if btc_close is not None and gold_close is not None:
        macro_regime, rolling_corr = detect_macro_regime(btc_close, gold_close)
        # Align macro regime to current DataFrame index
        result["macro_regime"] = macro_regime.reindex(df.index, method="ffill")
        result["btc_gold_corr"] = rolling_corr.reindex(df.index, method="ffill")

        # Encode macro regime as numeric
        macro_map = {
            MacroRegime.RISK_OFF: -1,
            MacroRegime.NEUTRAL: 0,
            MacroRegime.RISK_ON: 1,
            MacroRegime.DIVERGENT: 0,
        }
        result["macro_regime_numeric"] = result["macro_regime"].map(macro_map)

    logger.debug("Added regime detection features")
    return result


def _compute_adx(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX) for trend strength measurement.
    ADX above 25 indicates a trending market; below 25 indicates ranging.

    Uses close price approximation (simplified calculation).
    For full accuracy, high/low would be needed.

    Args:
        close: Close price series
        period: ADX calculation period

    Returns:
        ADX series (values 0-100)
    """
    # Simplified ADX using price changes as a proxy
    up_move = close.diff()
    down_move = -close.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed directional movement
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / close
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / close

    # Directional Index
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0)

    # ADX = smoothed DX
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx
