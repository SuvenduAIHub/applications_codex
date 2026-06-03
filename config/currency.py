"""
Multi-currency configuration for the trading system.
Supports INR and USD/USDT as base currencies for deposits and display.
Handles real-time exchange rate conversion between currencies.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests
from loguru import logger


class BaseCurrency(Enum):
    """Supported base currencies for the trading portfolio."""
    USD = "USD"     # US Dollar (default, used for USDT trading)
    INR = "INR"     # Indian Rupee (used for INR-based exchanges)


@dataclass
class CurrencyConfig:
    """Configuration for multi-currency support."""
    # User's selected base currency for deposits and display
    base_currency: BaseCurrency = BaseCurrency.USD

    # Currency symbols for display formatting
    symbols: dict = None

    # Exchange rate cache TTL in seconds (refresh every 5 minutes)
    rate_cache_ttl_seconds: int = 300

    def __post_init__(self):
        """Initialize currency display symbols."""
        if self.symbols is None:
            self.symbols = {
                BaseCurrency.USD: "$",
                BaseCurrency.INR: "₹",
            }

    @property
    def symbol(self) -> str:
        """Get the display symbol for the current base currency."""
        return self.symbols.get(self.base_currency, "$")


class CurrencyConverter:
    """
    Real-time currency converter using free open-source exchange rate APIs.
    Caches rates to minimize API calls. Falls back to a default rate if API fails.
    """

    # Free exchange rate API (no key required, open-source)
    RATE_API_URL = "https://open.er-api.com/v6/latest/{base}"

    # Fallback INR/USD rate if API is unavailable
    FALLBACK_INR_PER_USD = 83.50

    def __init__(self):
        """Initialize the converter with empty rate cache."""
        self._rates = {}         # Cached exchange rates
        self._last_fetch = 0     # Timestamp of last API call
        self._cache_ttl = 300    # Cache TTL in seconds (5 min)

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get the exchange rate from one currency to another.

        Args:
            from_currency: Source currency code (e.g., "USD")
            to_currency: Target currency code (e.g., "INR")

        Returns:
            Exchange rate (multiply amount in from_currency to get to_currency)
        """
        # Same currency returns 1.0
        if from_currency.upper() == to_currency.upper():
            return 1.0

        import time
        now = time.time()

        # Refresh rates if cache expired
        if not self._rates or (now - self._last_fetch > self._cache_ttl):
            self._fetch_rates()

        # Look up the rate
        key = f"{from_currency.upper()}_{to_currency.upper()}"
        if key in self._rates:
            return self._rates[key]

        # Try inverse rate
        inverse_key = f"{to_currency.upper()}_{from_currency.upper()}"
        if inverse_key in self._rates:
            return 1.0 / self._rates[inverse_key]

        # Fallback for common pairs
        if "USD" in key and "INR" in key:
            rate = self.FALLBACK_INR_PER_USD
            if from_currency.upper() == "INR":
                return 1.0 / rate
            return rate

        logger.warning(f"No rate found for {key}, returning 1.0")
        return 1.0

    def _fetch_rates(self):
        """
        Fetch latest exchange rates from the open API.
        Updates the internal rate cache on success.
        """
        import time

        try:
            # Fetch USD-based rates
            response = requests.get(
                self.RATE_API_URL.format(base="USD"),
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                rates = data.get("rates", {})

                # Cache commonly needed pairs
                if "INR" in rates:
                    self._rates["USD_INR"] = rates["INR"]
                    self._rates["INR_USD"] = 1.0 / rates["INR"]

                self._last_fetch = time.time()
                logger.info(f"Exchange rates updated: USD/INR = {rates.get('INR', 'N/A')}")
            else:
                logger.warning(f"Rate API returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to fetch exchange rates: {e}. Using fallback rates.")
            # Set fallback rates
            self._rates["USD_INR"] = self.FALLBACK_INR_PER_USD
            self._rates["INR_USD"] = 1.0 / self.FALLBACK_INR_PER_USD
            import time
            self._last_fetch = time.time()

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convert an amount from one currency to another.

        Args:
            amount: The amount to convert
            from_currency: Source currency code
            to_currency: Target currency code

        Returns:
            Converted amount in the target currency
        """
        rate = self.get_rate(from_currency, to_currency)
        return amount * rate

    def format_amount(self, amount: float, currency: BaseCurrency) -> str:
        """
        Format an amount with the appropriate currency symbol.

        Args:
            amount: Numeric amount
            currency: Currency for display formatting

        Returns:
            Formatted string (e.g., "$1,234.56" or "₹1,03,456.78")
        """
        symbol = "$" if currency == BaseCurrency.USD else "₹"

        if currency == BaseCurrency.INR:
            # Indian numbering format (lakhs, crores)
            return f"{symbol}{amount:,.2f}"
        return f"{symbol}{amount:,.2f}"

    def get_inr_usd_rate(self) -> float:
        """Convenience method to get current INR per USD rate."""
        return self.get_rate("USD", "INR")
