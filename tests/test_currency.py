"""
Unit tests for the multi-currency module.
Tests CurrencyConverter (rate fetching, caching, fallback, conversion)
and Portfolio dual-currency get_summary output.
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from config.currency import BaseCurrency, CurrencyConfig, CurrencyConverter
from src.execution.portfolio import Portfolio


# ======================================================================
# CurrencyConfig Tests
# ======================================================================

class TestCurrencyConfig:
    """Tests for the CurrencyConfig dataclass."""

    def test_default_base_currency(self):
        """Default base currency should be USD."""
        config = CurrencyConfig()
        assert config.base_currency == BaseCurrency.USD

    def test_inr_base_currency(self):
        """Should accept INR as base currency."""
        config = CurrencyConfig(base_currency=BaseCurrency.INR)
        assert config.base_currency == BaseCurrency.INR

    def test_usd_symbol(self):
        """USD should return $ symbol."""
        config = CurrencyConfig(base_currency=BaseCurrency.USD)
        assert config.symbol == "$"

    def test_inr_symbol(self):
        """INR should return Rupee symbol."""
        config = CurrencyConfig(base_currency=BaseCurrency.INR)
        assert config.symbol == "₹"

    def test_default_cache_ttl(self):
        """Default rate cache TTL should be 300 seconds (5 minutes)."""
        config = CurrencyConfig()
        assert config.rate_cache_ttl_seconds == 300


# ======================================================================
# CurrencyConverter Tests
# ======================================================================

class TestCurrencyConverter:
    """Tests for CurrencyConverter with mocked API responses."""

    def test_same_currency_rate(self):
        """Same currency conversion should return 1.0 without API call."""
        converter = CurrencyConverter()
        assert converter.get_rate("USD", "USD") == 1.0
        assert converter.get_rate("INR", "INR") == 1.0

    def test_same_currency_case_insensitive(self):
        """Currency codes should be case-insensitive."""
        converter = CurrencyConverter()
        assert converter.get_rate("usd", "USD") == 1.0
        assert converter.get_rate("Inr", "inr") == 1.0

    @patch("config.currency.requests.get")
    def test_fetch_usd_to_inr_from_api(self, mock_get):
        """Should fetch and return real rate from API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rates": {"INR": 84.25, "EUR": 0.92, "GBP": 0.79}
        }
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        rate = converter.get_rate("USD", "INR")
        assert rate == 84.25

    @patch("config.currency.requests.get")
    def test_fetch_inr_to_usd_inverse(self, mock_get):
        """INR to USD should return the inverse of USD to INR rate."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rates": {"INR": 84.0}
        }
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        rate = converter.get_rate("INR", "USD")
        assert rate == pytest.approx(1.0 / 84.0, rel=1e-6)

    @patch("config.currency.requests.get")
    def test_api_failure_uses_fallback(self, mock_get):
        """Should use fallback rate (83.50) when API fails."""
        mock_get.side_effect = Exception("Connection timeout")

        converter = CurrencyConverter()
        rate = converter.get_rate("USD", "INR")
        assert rate == CurrencyConverter.FALLBACK_INR_PER_USD

    @patch("config.currency.requests.get")
    def test_api_failure_fallback_inr_to_usd(self, mock_get):
        """Fallback INR->USD should be inverse of fallback rate."""
        mock_get.side_effect = Exception("Network error")

        converter = CurrencyConverter()
        rate = converter.get_rate("INR", "USD")
        assert rate == pytest.approx(1.0 / 83.50, rel=1e-6)

    @patch("config.currency.requests.get")
    def test_caching_prevents_repeated_api_calls(self, mock_get):
        """Second call within cache TTL should not trigger another API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rates": {"INR": 84.0}}
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        converter._cache_ttl = 300  # 5 min cache

        # First call fetches from API
        rate1 = converter.get_rate("USD", "INR")
        # Second call should use cache
        rate2 = converter.get_rate("USD", "INR")

        assert rate1 == rate2 == 84.0
        # API should only be called once due to caching
        assert mock_get.call_count == 1

    @patch("config.currency.requests.get")
    def test_cache_expires_after_ttl(self, mock_get):
        """Should re-fetch rates after cache TTL expires."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rates": {"INR": 84.0}}
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        converter._cache_ttl = 0  # Expire immediately

        converter.get_rate("USD", "INR")
        converter._last_fetch = 0  # Force cache expiry
        converter.get_rate("USD", "INR")

        # Should have fetched twice since cache expired
        assert mock_get.call_count == 2

    @patch("config.currency.requests.get")
    def test_convert_amount(self, mock_get):
        """convert() should multiply amount by exchange rate."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rates": {"INR": 84.0}}
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        result = converter.convert(1000.0, "USD", "INR")
        assert result == pytest.approx(84000.0, rel=1e-6)

    @patch("config.currency.requests.get")
    def test_convert_zero_amount(self, mock_get):
        """Converting 0 should return 0."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rates": {"INR": 84.0}}
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        assert converter.convert(0.0, "USD", "INR") == 0.0

    def test_format_amount_usd(self):
        """USD formatting should use $ symbol."""
        converter = CurrencyConverter()
        result = converter.format_amount(1234.56, BaseCurrency.USD)
        assert result.startswith("$")
        assert "1,234.56" in result

    def test_format_amount_inr(self):
        """INR formatting should use Rupee symbol."""
        converter = CurrencyConverter()
        result = converter.format_amount(103456.78, BaseCurrency.INR)
        assert "₹" in result

    @patch("config.currency.requests.get")
    def test_get_inr_usd_rate_convenience(self, mock_get):
        """get_inr_usd_rate() should return USD to INR rate."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rates": {"INR": 84.0}}
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        rate = converter.get_inr_usd_rate()
        assert rate == 84.0

    @patch("config.currency.requests.get")
    def test_api_non_200_uses_fallback(self, mock_get):
        """Non-200 API response should trigger fallback rates."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        # After a non-200 response, the fallback should kick in
        rate = converter.get_rate("USD", "INR")
        assert rate == CurrencyConverter.FALLBACK_INR_PER_USD

    def test_unknown_currency_pair_returns_one(self):
        """Unknown currency pair without fallback should return 1.0."""
        converter = CurrencyConverter()
        # Pre-populate cache to avoid API call
        converter._rates = {"USD_INR": 84.0}
        converter._last_fetch = time.time()

        rate = converter.get_rate("USD", "JPY")
        assert rate == 1.0


# ======================================================================
# Portfolio Dual-Currency Summary Tests
# ======================================================================

class TestPortfolioDualCurrency:
    """Tests for Portfolio.get_summary() dual-currency output."""

    def test_summary_without_inr_rate(self):
        """Summary with no INR rate should not include INR or USD sub-objects."""
        portfolio = Portfolio(initial_balance=100000.0)
        summary = portfolio.get_summary()
        assert "inr" not in summary
        assert "usd" not in summary
        assert summary["currency"] == "USD"

    def test_summary_with_zero_inr_rate(self):
        """INR rate of 0 should not include INR values (same as no rate)."""
        portfolio = Portfolio(initial_balance=100000.0)
        summary = portfolio.get_summary(inr_rate=0.0)
        assert "inr" not in summary
        assert "usd" not in summary

    def test_summary_with_inr_rate(self):
        """Positive INR rate should include both INR and USD sub-objects."""
        portfolio = Portfolio(initial_balance=100000.0)
        summary = portfolio.get_summary(inr_rate=84.0)

        # INR sub-object should exist with converted values
        assert "inr" in summary
        assert summary["inr"]["total_value"] == pytest.approx(100000.0 * 84.0)
        assert summary["inr"]["cash_balance"] == pytest.approx(100000.0 * 84.0)
        assert summary["inr"]["initial_balance"] == pytest.approx(100000.0 * 84.0)
        assert summary["inr"]["exchange_rate"] == 84.0

        # USD sub-object should contain original values
        assert "usd" in summary
        assert summary["usd"]["total_value"] == 100000.0
        assert summary["usd"]["cash_balance"] == 100000.0

    def test_summary_inr_pnl_conversion(self):
        """INR PnL values should be correctly converted."""
        portfolio = Portfolio(initial_balance=100000.0)
        # Simulate a closed trade with $500 PnL via trade_history
        portfolio.trade_history.append({"pnl": 500.0, "symbol": "BTC/USDT"})
        summary = portfolio.get_summary(inr_rate=84.0)

        assert summary["inr"]["total_realized_pnl"] == pytest.approx(500.0 * 84.0)
        assert summary["usd"]["total_realized_pnl"] == 500.0

    def test_summary_top_level_fields_unchanged(self):
        """Top-level fields should remain in USD regardless of INR rate."""
        portfolio = Portfolio(initial_balance=50000.0)
        summary = portfolio.get_summary(inr_rate=84.0)

        # Top-level values should always be in USD
        assert summary["total_value"] == 50000.0
        assert summary["cash_balance"] == 50000.0
        assert summary["initial_balance"] == 50000.0

    def test_summary_total_trades_count(self):
        """Total trades should reflect trade history length."""
        portfolio = Portfolio(initial_balance=100000.0)
        assert portfolio.get_summary()["total_trades"] == 0

    def test_summary_open_positions_empty(self):
        """Open positions should be empty dict when no positions held."""
        portfolio = Portfolio(initial_balance=100000.0)
        assert portfolio.get_summary()["open_positions"] == {}
