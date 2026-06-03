"""
Trading pair definitions and metadata for BTC/USDT and XAU/USD.
Contains pair-specific parameters like tick sizes, lot sizes, and trading hours.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TradingPairInfo:
    """Metadata and trading rules for a specific trading pair."""
    # Unique pair symbol (e.g., "BTC/USDT")
    symbol: str

    # Base asset (e.g., "BTC", "XAU")
    base_asset: str

    # Quote asset (e.g., "USDT", "USD")
    quote_asset: str

    # Minimum price increment (tick size)
    tick_size: float

    # Minimum trade quantity
    min_quantity: float

    # Maximum trade quantity
    max_quantity: float

    # Quantity step (lot size increment)
    quantity_step: float

    # Price decimal precision
    price_precision: int

    # Quantity decimal precision
    quantity_precision: int

    # Is the market open 24/7 or follows trading hours
    is_24_7: bool

    # Data source identifier for the feed provider
    feed_symbol: str

    # Human-readable description
    description: str

    # Asset class category
    asset_class: str


# BTC/USDT pair configuration - trades on crypto exchanges (24/7)
BTC_USDT = TradingPairInfo(
    symbol="BTC/USDT",
    base_asset="BTC",
    quote_asset="USDT",
    tick_size=0.01,           # Minimum price change: $0.01
    min_quantity=0.00001,     # Minimum order: 0.00001 BTC
    max_quantity=100.0,       # Maximum order: 100 BTC
    quantity_step=0.00001,    # Order increment: 0.00001 BTC
    price_precision=2,        # Price shown to 2 decimals
    quantity_precision=5,     # Quantity to 5 decimals
    is_24_7=True,             # Crypto trades 24/7
    feed_symbol="BTCUSDT",   # Binance symbol format
    description="Bitcoin vs Tether USD",
    asset_class="cryptocurrency",
)

# XAU/USD pair configuration - trades on forex markets (weekday hours)
XAU_USD = TradingPairInfo(
    symbol="XAU/USD",
    base_asset="XAU",
    quote_asset="USD",
    tick_size=0.01,           # Minimum price change: $0.01
    min_quantity=0.01,        # Minimum order: 0.01 troy oz
    max_quantity=1000.0,      # Maximum order: 1000 troy oz
    quantity_step=0.01,       # Order increment: 0.01 troy oz
    price_precision=2,        # Price shown to 2 decimals
    quantity_precision=2,     # Quantity to 2 decimals
    is_24_7=False,            # Gold follows forex trading hours (Sun 5pm - Fri 5pm ET)
    feed_symbol="GC=F",      # Yahoo Finance futures symbol for Gold
    description="Gold Spot vs US Dollar",
    asset_class="commodity",
)


def get_trading_pairs() -> Dict[str, TradingPairInfo]:
    """Return a dictionary of all configured trading pairs."""
    return {
        "BTC/USDT": BTC_USDT,
        "XAU/USD": XAU_USD,
    }


def get_pair_info(symbol: str) -> Optional[TradingPairInfo]:
    """Look up trading pair metadata by symbol."""
    pairs = get_trading_pairs()
    return pairs.get(symbol)
