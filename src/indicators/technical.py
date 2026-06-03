"""
Technical analysis indicators for the trading system.
Computes RSI, MACD, EMA, Bollinger Bands, ATR, Supertrend, Ichimoku Cloud,
Parabolic SAR, Stochastic RSI, ROC, Keltner Channels, Donchian Channels,
and volume-based features. All indicators operate on pandas DataFrames/Series.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Exponential Moving Average (EMA).

    Args:
        series: Price series (typically close prices)
        period: EMA lookback period

    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Simple Moving Average (SMA).

    Args:
        series: Price series
        period: SMA lookback period

    Returns:
        SMA series
    """
    return series.rolling(window=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    RSI measures momentum by comparing average gains to average losses.

    Args:
        series: Price series (typically close prices)
        period: RSI lookback period (default 14)

    Returns:
        RSI series (values between 0 and 100)
    """
    delta = series.diff()
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Use exponential moving average for smoothing (Wilder's method)
    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)  # Default to neutral (50) when data is insufficient


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Compute Moving Average Convergence Divergence (MACD).
    Returns MACD line, signal line, and histogram.

    Args:
        series: Price series (typically close prices)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        DataFrame with columns: [macd, macd_signal, macd_histogram]
    """
    ema_fast = compute_ema(series, fast_period)
    ema_slow = compute_ema(series, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    })


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands (upper, middle, lower).
    Bollinger Bands measure volatility and identify overbought/oversold conditions.

    Args:
        series: Price series (typically close prices)
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        DataFrame with columns: [bb_upper, bb_middle, bb_lower, bb_width, bb_pct]
    """
    middle = compute_sma(series, period)
    rolling_std = series.rolling(window=period).std()

    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)

    # Band width: normalized width of the bands
    width = (upper - lower) / middle

    # %B: position of price relative to bands (0 = lower, 1 = upper)
    pct_b = (series - lower) / (upper - lower)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct": pct_b,
    })


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR).
    ATR measures market volatility using the range of price movement.

    Args:
        high: High prices series
        low: Low prices series
        close: Close prices series
        period: ATR lookback period (default 14)

    Returns:
        ATR series
    """
    # True Range is the maximum of three values
    tr1 = high - low                          # Current high - current low
    tr2 = (high - close.shift(1)).abs()       # Current high - previous close
    tr3 = (low - close.shift(1)).abs()        # Current low - previous close

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the exponential moving average of true range
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_volume_features(
    volume: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """
    Compute volume-based features for analysis.
    Includes volume MA, volume ratio, OBV, and VWAP approximation.

    Args:
        volume: Volume series
        close: Close price series
        period: Lookback period for moving averages

    Returns:
        DataFrame with volume features
    """
    # Volume moving average
    vol_ma = compute_sma(volume, period)

    # Volume ratio (current volume vs average)
    vol_ratio = volume / vol_ma.replace(0, np.nan)

    # On-Balance Volume (OBV) - cumulative volume indicator
    price_change = close.diff()
    obv_direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = (volume * obv_direction).cumsum()

    # Volume-weighted price (simplified VWAP proxy)
    vwap = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

    return pd.DataFrame({
        "volume_ma": vol_ma,
        "volume_ratio": vol_ratio,
        "obv": obv,
        "vwap": vwap,
    })


def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average Directional Index (ADX) to measure trend strength.
    ADX > 25 indicates a strong trend, > 40 is very strong.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period (default 14)

    Returns:
        ADX values as a Series
    """
    # +DM and -DM (directional movement)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, min_periods=period).mean()

    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

    return adx.fillna(0)


def compute_volatility_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """
    Compute volatility-based features.
    Includes historical volatility, Garman-Klass volatility, and normalized ATR.

    Args:
        close: Close price series
        high: High price series
        low: Low price series
        period: Lookback period

    Returns:
        DataFrame with volatility features
    """
    # Log returns for volatility calculation
    log_returns = np.log(close / close.shift(1))

    # Historical (realized) volatility: annualized standard deviation of log returns
    hist_vol = log_returns.rolling(window=period).std() * np.sqrt(252)

    # Garman-Klass volatility estimator (more efficient than close-to-close)
    gk_vol = np.sqrt(
        (0.5 * np.log(high / low) ** 2
         - (2 * np.log(2) - 1) * np.log(close / close.shift(1)) ** 2
         ).rolling(window=period).mean() * 252
    )

    # Normalized ATR (ATR as percentage of price)
    atr = compute_atr(high, low, close, period)
    norm_atr = (atr / close) * 100

    # Returns skewness (measures asymmetry of returns distribution)
    returns_skew = log_returns.rolling(window=period).skew()

    return pd.DataFrame({
        "hist_volatility": hist_vol,
        "gk_volatility": gk_vol,
        "norm_atr_pct": norm_atr,
        "returns_skew": returns_skew,
    })


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Compute Supertrend indicator — a trend-following overlay that flips
    between bullish (price above band) and bearish (price below band).
    Uses ATR to set dynamic support/resistance levels.

    Args:
        high, low, close: OHLC price series
        period: ATR lookback period (default 10)
        multiplier: ATR multiplier for band distance (default 3.0)

    Returns:
        DataFrame with columns: [supertrend, supertrend_direction]
        direction: 1 = bullish (uptrend), -1 = bearish (downtrend)
    """
    atr = compute_atr(high, low, close, period)
    hl2 = (high + low) / 2

    # Basic upper and lower bands
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Initialize final bands
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)

    for i in range(1, len(close)):
        # Upper band: take lower value if previous close was above previous upper
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band: take higher value if previous close was below previous lower
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    # Determine direction and supertrend value
    for i in range(len(close)):
        if i == 0:
            direction.iloc[i] = 1
            supertrend.iloc[i] = final_lower.iloc[i]
            continue

        if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            # Was bearish — switch to bullish if close breaks above upper
            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
        else:
            # Was bullish — switch to bearish if close breaks below lower
            if close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]

    return pd.DataFrame({
        "supertrend": supertrend,
        "supertrend_direction": direction,
    })


def compute_ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> pd.DataFrame:
    """
    Compute Ichimoku Cloud components — a complete trend system that shows
    support/resistance, trend direction, and momentum in a single view.

    Args:
        high, low, close: OHLC price series
        tenkan_period: Conversion line period (default 9)
        kijun_period: Base line period (default 26)
        senkou_b_period: Leading span B period (default 52)

    Returns:
        DataFrame with: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    """
    # Tenkan-sen (Conversion Line): midpoint of 9-period high-low
    tenkan_sen = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2

    # Kijun-sen (Base Line): midpoint of 26-period high-low
    kijun_sen = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2

    # Senkou Span A (Leading Span A): midpoint of tenkan and kijun, shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

    # Senkou Span B (Leading Span B): midpoint of 52-period high-low, shifted forward 26 periods
    senkou_span_b = ((high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2).shift(kijun_period)

    # Chikou Span (Lagging Span): close shifted back 26 periods
    chikou_span = close.shift(-kijun_period)

    return pd.DataFrame({
        "ichimoku_tenkan": tenkan_sen,
        "ichimoku_kijun": kijun_sen,
        "ichimoku_senkou_a": senkou_span_a,
        "ichimoku_senkou_b": senkou_span_b,
        "ichimoku_chikou": chikou_span,
    })


def compute_parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.DataFrame:
    """
    Compute Parabolic SAR — a trailing stop/reversal indicator.
    Dots below price = uptrend, dots above price = downtrend.

    Args:
        high, low, close: OHLC price series
        af_start: Initial acceleration factor (default 0.02)
        af_step: AF increment on new highs/lows (default 0.02)
        af_max: Maximum acceleration factor (default 0.20)

    Returns:
        DataFrame with: psar, psar_direction (1=bullish, -1=bearish)
    """
    length = len(close)
    psar = close.copy()
    direction = pd.Series(1.0, index=close.index)
    af = af_start
    ep = low.iloc[0]  # Extreme point
    hp = high.iloc[0]
    lp = low.iloc[0]

    for i in range(2, length):
        if direction.iloc[i - 1] == 1:  # Uptrend
            psar.iloc[i] = psar.iloc[i - 1] + af * (hp - psar.iloc[i - 1])
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i - 1], low.iloc[i - 2])

            if low.iloc[i] < psar.iloc[i]:
                # Reversal to downtrend
                direction.iloc[i] = -1
                psar.iloc[i] = hp
                lp = low.iloc[i]
                af = af_start
            else:
                direction.iloc[i] = 1
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:  # Downtrend
            psar.iloc[i] = psar.iloc[i - 1] + af * (lp - psar.iloc[i - 1])
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i - 1], high.iloc[i - 2])

            if high.iloc[i] > psar.iloc[i]:
                # Reversal to uptrend
                direction.iloc[i] = 1
                psar.iloc[i] = lp
                hp = high.iloc[i]
                af = af_start
            else:
                direction.iloc[i] = -1
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + af_step, af_max)

    return pd.DataFrame({
        "psar": psar,
        "psar_direction": direction,
    })


def compute_stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """
    Compute Stochastic RSI — applies stochastic oscillator formula to RSI values.
    More sensitive than plain RSI for detecting fast reversals.

    Args:
        close: Close price series
        rsi_period: RSI calculation period (default 14)
        stoch_period: Stochastic lookback on RSI (default 14)
        smooth_k: %K smoothing period (default 3)
        smooth_d: %D smoothing period (default 3)

    Returns:
        DataFrame with: stoch_rsi_k, stoch_rsi_d (both 0-100 range)
    """
    rsi = compute_rsi(close, rsi_period)

    # Apply stochastic formula to RSI
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    stoch_rsi = stoch_rsi.fillna(0.5) * 100  # Scale to 0-100

    # Smooth %K and %D
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()

    return pd.DataFrame({
        "stoch_rsi_k": k,
        "stoch_rsi_d": d,
    })


def compute_roc(close: pd.Series, period: int = 12) -> pd.Series:
    """
    Compute Rate of Change (ROC) — momentum indicator measuring
    percentage change over N periods. Positive = upward momentum.

    Args:
        close: Close price series
        period: Lookback period (default 12)

    Returns:
        ROC series (percentage values)
    """
    return ((close - close.shift(period)) / close.shift(period).replace(0, np.nan)) * 100


def compute_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Compute Keltner Channels — EMA-based bands using ATR for width.
    Used with Bollinger Bands for squeeze detection: when BB is inside
    Keltner, volatility is compressed and a breakout is imminent.

    Args:
        high, low, close: OHLC price series
        ema_period: Center line EMA period (default 20)
        atr_period: ATR period for band width (default 10)
        multiplier: ATR multiplier (default 1.5)

    Returns:
        DataFrame with: keltner_upper, keltner_middle, keltner_lower
    """
    middle = compute_ema(close, ema_period)
    atr = compute_atr(high, low, close, atr_period)

    return pd.DataFrame({
        "keltner_upper": middle + (multiplier * atr),
        "keltner_middle": middle,
        "keltner_lower": middle - (multiplier * atr),
    })


def compute_donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """
    Compute Donchian Channels — highest high and lowest low over N periods.
    Classic breakout indicator: price breaking above upper = bullish breakout.

    Args:
        high, low: High and low price series
        period: Lookback period (default 20)

    Returns:
        DataFrame with: donchian_upper, donchian_middle, donchian_lower
    """
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2

    return pd.DataFrame({
        "donchian_upper": upper,
        "donchian_middle": middle,
        "donchian_lower": lower,
    })


def compute_squeeze_indicator(
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    keltner_upper: pd.Series,
    keltner_lower: pd.Series,
) -> pd.Series:
    """
    Compute BB + Keltner Squeeze detection.
    When Bollinger Bands are INSIDE Keltner Channels, the market is in a
    low-volatility squeeze. A breakout from squeeze often produces large moves.

    Args:
        bb_upper, bb_lower: Bollinger Band boundaries
        keltner_upper, keltner_lower: Keltner Channel boundaries

    Returns:
        Series: True when squeeze is active (BB inside Keltner)
    """
    squeeze_on = (bb_lower > keltner_lower) & (bb_upper < keltner_upper)
    return squeeze_on


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Williams %R — momentum oscillator measuring overbought/oversold.
    Range: -100 to 0. Values near -100 = oversold, near 0 = overbought.
    Similar to Stochastic but inverted scale and uses only close (no smoothing).

    Args:
        high, low, close: OHLC price series
        period: Lookback period (default 14)

    Returns:
        Williams %R series (-100 to 0)
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return wr.fillna(-50)


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Compute Commodity Channel Index (CCI) — measures price deviation from
    its statistical mean. Values above +100 = overbought/strong uptrend,
    below -100 = oversold/strong downtrend. Useful for Gold trading.

    Args:
        high, low, close: OHLC price series
        period: Lookback period (default 20)

    Returns:
        CCI series (unbounded, typically -300 to +300)
    """
    # Typical Price = (High + Low + Close) / 3
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(period).mean()
    # Mean Deviation = average of absolute deviations from SMA
    mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    # CCI = (TP - SMA) / (0.015 * Mean Deviation)
    cci = (tp - tp_sma) / (0.015 * mean_dev).replace(0, np.nan)
    return cci.fillna(0)


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Money Flow Index (MFI) — volume-weighted RSI.
    Combines price AND volume to measure buying/selling pressure.
    Above 80 = overbought, below 20 = oversold.

    Args:
        high, low, close: OHLC price series
        volume: Volume series
        period: Lookback period (default 14)

    Returns:
        MFI series (0-100)
    """
    # Typical Price
    tp = (high + low + close) / 3
    # Raw Money Flow = TP * Volume
    raw_mf = tp * volume
    # Positive/Negative money flow based on TP direction
    tp_diff = tp.diff()
    pos_mf = pd.Series(np.where(tp_diff > 0, raw_mf, 0), index=close.index)
    neg_mf = pd.Series(np.where(tp_diff < 0, raw_mf, 0), index=close.index)
    # Money Flow Ratio
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mf_ratio = pos_sum / neg_sum.replace(0, np.nan)
    # MFI = 100 - (100 / (1 + MFR))
    mfi = 100 - (100 / (1 + mf_ratio))
    return mfi.fillna(50)


def compute_pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Compute Pivot Points — key intraday support/resistance levels
    derived from previous period's High, Low, Close.
    Widely used by institutional traders for entry/exit levels.

    Args:
        high, low, close: OHLC price series

    Returns:
        DataFrame with: pivot, pivot_r1, pivot_r2, pivot_s1, pivot_s2
        (pivot = center, R1/R2 = resistance, S1/S2 = support)
    """
    # Use previous candle's HLC for pivot calculation
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # Standard pivot point formula
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = (2 * pivot) - prev_low       # First resistance
    s1 = (2 * pivot) - prev_high       # First support
    r2 = pivot + (prev_high - prev_low)  # Second resistance
    s2 = pivot - (prev_high - prev_low)  # Second support

    return pd.DataFrame({
        "pivot": pivot,
        "pivot_r1": r1,
        "pivot_r2": r2,
        "pivot_s1": s1,
        "pivot_s2": s2,
    })


def compute_accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Compute Accumulation/Distribution Line — measures the cumulative flow
    of money into/out of an asset. Rising A/D = accumulation (buying),
    falling A/D = distribution (selling). Divergence with price = reversal signal.

    Args:
        high, low, close: OHLC price series
        volume: Volume series

    Returns:
        A/D line series (cumulative)
    """
    # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfm = mfm.fillna(0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # A/D Line = cumulative sum of MFV
    return mfv.cumsum()


def add_all_indicators(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Add all technical indicators and features to an OHLCV DataFrame.
    This is the main entry point for feature engineering.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        config: Optional StrategyConfig for custom parameters

    Returns:
        Enhanced DataFrame with all indicator columns added
    """
    # Import here to avoid circular imports
    from config.settings import StrategyConfig
    cfg = config or StrategyConfig()

    result = df.copy()

    # --- Moving Averages ---
    result["ema_fast"] = compute_ema(df["close"], cfg.ema_fast_period)
    result["ema_slow"] = compute_ema(df["close"], cfg.ema_slow_period)
    result["sma_20"] = compute_sma(df["close"], 20)
    result["sma_50"] = compute_sma(df["close"], 50)
    result["sma_200"] = compute_sma(df["close"], 200)

    # --- RSI ---
    result["rsi"] = compute_rsi(df["close"], cfg.rsi_period)

    # --- MACD ---
    macd_df = compute_macd(df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    result = pd.concat([result, macd_df], axis=1)

    # --- Bollinger Bands ---
    bb_df = compute_bollinger_bands(df["close"], cfg.bb_period, cfg.bb_std_dev)
    result = pd.concat([result, bb_df], axis=1)

    # --- ATR ---
    result["atr"] = compute_atr(df["high"], df["low"], df["close"], cfg.atr_period)

    # --- ADX (Average Directional Index) for trend strength ---
    result["adx"] = _compute_adx(df["high"], df["low"], df["close"], period=14)

    # --- Volume Features ---
    vol_df = compute_volume_features(df["volume"], df["close"], cfg.volume_ma_period)
    result = pd.concat([result, vol_df], axis=1)

    # --- Volatility Features ---
    vola_df = compute_volatility_features(
        df["close"], df["high"], df["low"], period=20
    )
    result = pd.concat([result, vola_df], axis=1)

    # --- Supertrend (trend-following overlay) ---
    st_df = compute_supertrend(df["high"], df["low"], df["close"])
    result = pd.concat([result, st_df], axis=1)

    # --- Ichimoku Cloud (complete trend system) ---
    ichi_df = compute_ichimoku(df["high"], df["low"], df["close"])
    result = pd.concat([result, ichi_df], axis=1)

    # --- Parabolic SAR (trailing stop/reversal) ---
    psar_df = compute_parabolic_sar(df["high"], df["low"], df["close"])
    result = pd.concat([result, psar_df], axis=1)

    # --- Stochastic RSI (fast reversal detection) ---
    stoch_rsi_df = compute_stochastic_rsi(df["close"])
    result = pd.concat([result, stoch_rsi_df], axis=1)

    # --- ROC (Rate of Change momentum) ---
    result["roc"] = compute_roc(df["close"], period=12)

    # --- Keltner Channels (for BB squeeze detection) ---
    kelt_df = compute_keltner_channels(df["high"], df["low"], df["close"])
    result = pd.concat([result, kelt_df], axis=1)

    # --- Donchian Channels (breakout detection) ---
    donch_df = compute_donchian_channels(df["high"], df["low"], period=20)
    result = pd.concat([result, donch_df], axis=1)

    # --- BB + Keltner Squeeze detection ---
    result["squeeze_on"] = compute_squeeze_indicator(
        result["bb_upper"], result["bb_lower"],
        result["keltner_upper"], result["keltner_lower"],
    )

    # --- EMA 200 (long-term trend reference) ---
    result["ema_200"] = compute_ema(df["close"], 200)

    # --- Williams %R (momentum oscillator, -100 to 0) ---
    result["williams_r"] = compute_williams_r(df["high"], df["low"], df["close"])

    # --- CCI (Commodity Channel Index — great for Gold) ---
    result["cci"] = compute_cci(df["high"], df["low"], df["close"])

    # --- MFI (Money Flow Index — volume-weighted RSI) ---
    result["mfi"] = compute_mfi(df["high"], df["low"], df["close"], df["volume"])

    # --- Pivot Points (institutional support/resistance levels) ---
    pivot_df = compute_pivot_points(df["high"], df["low"], df["close"])
    result = pd.concat([result, pivot_df], axis=1)

    # --- Accumulation/Distribution Line (money flow tracking) ---
    result["ad_line"] = compute_accumulation_distribution(
        df["high"], df["low"], df["close"], df["volume"]
    )

    # --- Price-based Features ---
    # Returns over different periods
    result["return_1"] = df["close"].pct_change(1)
    result["return_5"] = df["close"].pct_change(5)
    result["return_10"] = df["close"].pct_change(10)

    # Price relative to moving averages
    result["price_vs_sma20"] = (df["close"] - result["sma_20"]) / result["sma_20"]
    result["price_vs_sma50"] = (df["close"] - result["sma_50"]) / result["sma_50"]

    # High-low range as percentage of close
    result["hl_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

    # Price vs VWAP (institutional bias — above VWAP = bullish, below = bearish)
    result["price_vs_vwap"] = (df["close"] - result["vwap"]) / result["vwap"]

    logger.debug(f"Added {len(result.columns) - len(df.columns)} indicator columns")
    return result
