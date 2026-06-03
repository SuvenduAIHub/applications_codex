"""
Correlation / Macro strategy — trades BTC and Gold based on macro indicators.
BTC and Gold react to USD strength, interest rates, inflation, and risk sentiment.

Uses DXY (Dollar Index), US10Y (Treasury yields), VIX (volatility index),
and S&P500 to determine risk-on/risk-off regime and generate directional signals.

External data fetched from Yahoo Finance (open-source, free):
    - ^DXY or DX-Y.NYB: US Dollar Index
    - ^TNX: US 10-Year Treasury Yield
    - ^VIX: CBOE Volatility Index
    - ^GSPC: S&P 500

Strategy logic:
    - Long BTC when: DXY falling + VIX low + S&P rising (risk-on environment)
    - Long Gold when: DXY falling + yields falling + VIX rising (flight to safety)
    - Short/avoid when: DXY rising strongly + yields rising (strong dollar, risk-off)
"""

import pandas as pd
import yfinance as yf
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal


class CorrelationMacroStrategy(BaseStrategy):
    """
    Macro-driven strategy using cross-asset correlations.
    Fetches DXY, VIX, US10Y, S&P500 data and generates
    signals based on inter-market relationships.
    """

    def __init__(self, config=None):
        super().__init__(config)
        # Cache macro data to avoid fetching on every call (refresh every 100 calls)
        self._macro_cache = None
        self._cache_counter = 0
        self._cache_interval = 100

    def get_name(self) -> str:
        return "correlation_macro"

    def _fetch_macro_data(self) -> dict:
        """
        Fetch macro indicators from Yahoo Finance.
        Returns dict with DXY trend, VIX level, yields trend, SPX trend.
        Caches results to avoid API rate limits.
        """
        self._cache_counter += 1
        if self._macro_cache and self._cache_counter < self._cache_interval:
            return self._macro_cache

        macro = {
            "dxy_falling": False, "dxy_rising": False,
            "vix_high": False, "vix_low": False, "vix_rising": False,
            "yields_falling": False, "yields_rising": False,
            "spx_rising": False, "spx_falling": False,
            "available": False,
        }

        try:
            # Fetch 5 days of daily data for trend detection
            tickers = yf.download(
                "DX-Y.NYB ^TNX ^VIX ^GSPC",
                period="10d", interval="1d",
                progress=False, auto_adjust=True,
            )

            if tickers.empty or len(tickers) < 3:
                logger.warning("Macro data unavailable — skipping correlation signals")
                self._macro_cache = macro
                return macro

            close = tickers["Close"] if "Close" in tickers.columns.get_level_values(0) else tickers

            # DXY (Dollar Index) — falling dollar = bullish for BTC and Gold
            if "DX-Y.NYB" in close.columns:
                dxy = close["DX-Y.NYB"].dropna()
                if len(dxy) >= 3:
                    dxy_change = (dxy.iloc[-1] - dxy.iloc[-3]) / dxy.iloc[-3]
                    macro["dxy_falling"] = dxy_change < -0.002
                    macro["dxy_rising"] = dxy_change > 0.002

            # VIX — high VIX = fear (good for Gold, mixed for BTC)
            if "^VIX" in close.columns:
                vix = close["^VIX"].dropna()
                if len(vix) >= 2:
                    macro["vix_high"] = float(vix.iloc[-1]) > 25
                    macro["vix_low"] = float(vix.iloc[-1]) < 18
                    macro["vix_rising"] = float(vix.iloc[-1]) > float(vix.iloc[-2])

            # US 10Y Yields — falling yields = bullish for Gold
            if "^TNX" in close.columns:
                tnx = close["^TNX"].dropna()
                if len(tnx) >= 3:
                    tnx_change = (tnx.iloc[-1] - tnx.iloc[-3]) / tnx.iloc[-3]
                    macro["yields_falling"] = tnx_change < -0.01
                    macro["yields_rising"] = tnx_change > 0.01

            # S&P 500 — rising SPX = risk-on (good for BTC)
            if "^GSPC" in close.columns:
                spx = close["^GSPC"].dropna()
                if len(spx) >= 3:
                    spx_change = (spx.iloc[-1] - spx.iloc[-3]) / spx.iloc[-3]
                    macro["spx_rising"] = spx_change > 0.003
                    macro["spx_falling"] = spx_change < -0.003

            macro["available"] = True
            logger.debug(f"Macro data: DXY_fall={macro['dxy_falling']}, VIX_high={macro['vix_high']}, "
                        f"yields_fall={macro['yields_falling']}, SPX_rise={macro['spx_rising']}")

        except Exception as e:
            logger.warning(f"Failed to fetch macro data: {e}")

        self._macro_cache = macro
        self._cache_counter = 0
        return macro

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate signal based on macro cross-asset correlations.
        Different logic for BTC vs Gold based on their macro sensitivities.
        """
        if len(df) < 10:
            return self._neutral_signal(df, symbol)

        curr = df.iloc[-1]
        price = curr["close"]
        atr = float(curr.get("atr", price * 0.02))

        # Fetch macro data
        macro = self._fetch_macro_data()
        if not macro["available"]:
            return self._neutral_signal(df, symbol)

        buy_factors = {}
        sell_factors = {}

        is_btc = "BTC" in symbol.upper()
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()

        if is_btc:
            # BTC: risk-on asset — thrives when dollar weak, stocks rising, fear low
            if macro["dxy_falling"]:
                buy_factors["dxy_weak"] = 0.7
            elif macro["dxy_rising"]:
                sell_factors["dxy_strong"] = 0.7

            if macro["vix_low"]:
                buy_factors["vix_calm"] = 0.6
            elif macro["vix_high"]:
                sell_factors["vix_fear"] = 0.5

            if macro["spx_rising"]:
                buy_factors["spx_risk_on"] = 0.7
            elif macro["spx_falling"]:
                sell_factors["spx_risk_off"] = 0.6

            # BTC benefits from falling yields (cheap money = risk appetite)
            if macro["yields_falling"]:
                buy_factors["yields_down"] = 0.5
            elif macro["yields_rising"]:
                sell_factors["yields_up"] = 0.5

        elif is_gold:
            # Gold: safe haven — thrives when dollar weak, yields falling, fear rising
            if macro["dxy_falling"]:
                buy_factors["dxy_weak"] = 0.8
            elif macro["dxy_rising"]:
                sell_factors["dxy_strong"] = 0.8

            if macro["yields_falling"]:
                buy_factors["yields_down"] = 0.8
            elif macro["yields_rising"]:
                sell_factors["yields_up"] = 0.7

            if macro["vix_rising"] or macro["vix_high"]:
                buy_factors["vix_fear"] = 0.7
            elif macro["vix_low"]:
                sell_factors["vix_calm"] = 0.4

            if macro["spx_falling"]:
                buy_factors["spx_flight_safety"] = 0.6
            elif macro["spx_rising"]:
                sell_factors["spx_risk_on"] = 0.4

        # Add technical confirmation from the price data
        # Price trend alignment (EMA fast vs slow)
        ema_fast = float(curr.get("ema_fast", price))
        ema_slow = float(curr.get("ema_slow", price))
        if ema_fast > ema_slow:
            buy_factors["ema_bullish"] = 0.4
        elif ema_fast < ema_slow:
            sell_factors["ema_bearish"] = 0.4

        buy_confidence = self._calculate_confidence(buy_factors)
        sell_confidence = self._calculate_confidence(sell_factors)

        min_factors = 2

        if len(buy_factors) >= min_factors and buy_confidence > sell_confidence:
            signal = Signal.STRONG_BUY if buy_confidence > 0.7 else Signal.BUY
            return TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=buy_confidence,
                strategy_name=self.get_name(),
                price=price,
                stop_loss=price - (2 * atr),
                take_profit=price + (3 * atr),
                metadata={
                    "macro": {k: v for k, v in macro.items() if k != "available"},
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
                stop_loss=price + (2 * atr),
                take_profit=price - (3 * atr),
                metadata={
                    "macro": {k: v for k, v in macro.items() if k != "available"},
                    "factors": sell_factors,
                },
            )

        return self._neutral_signal(df, symbol)
