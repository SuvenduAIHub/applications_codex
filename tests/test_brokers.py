"""
Unit tests for live broker integrations (Binance, WazirX, Delta Exchange, XM).
All exchange API calls are mocked — no real API keys or network required.
Tests cover order submission, balance fetching, price fetching, cancellation,
error handling, and execution statistics for each broker.
"""

import pytest
from unittest.mock import patch, MagicMock

from config.settings import OrderSide, OrderStatus, OrderType
from src.execution.order import Order
from src.execution.portfolio import Portfolio
from src.execution.live_broker import (
    BinanceLiveBroker,
    WazirXLiveBroker,
    DeltaExchangeLiveBroker,
    XMBroker,
)


# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def portfolio():
    """Create a fresh portfolio with $100k balance for each test."""
    return Portfolio(initial_balance=100000.0)


def _make_order(symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=0.01, price=50000.0, stop_price=None):
    """Helper to create a test Order object."""
    return Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        stop_price=stop_price,
    )


# ======================================================================
# Binance Live Broker Tests
# ======================================================================

class TestBinanceLiveBroker:
    """Tests for BinanceLiveBroker with mocked Binance API responses."""

    def _make_broker(self, portfolio, testnet=True):
        """Create a Binance broker instance for testing."""
        return BinanceLiveBroker(
            api_key="test_api_key",
            api_secret="test_api_secret",
            portfolio=portfolio,
            testnet=testnet,
        )

    def test_init_testnet(self, portfolio):
        """Testnet mode should use testnet base URL."""
        broker = self._make_broker(portfolio, testnet=True)
        assert "testnet" in broker.BASE_URL.lower() or broker.BASE_URL == "https://testnet.binance.vision"

    def test_init_live(self, portfolio):
        """Live mode should use production base URL."""
        broker = self._make_broker(portfolio, testnet=False)
        assert broker.BASE_URL == "https://api.binance.com"

    def test_sign_request_adds_timestamp_and_signature(self, portfolio):
        """Signed requests must include timestamp and HMAC signature."""
        broker = self._make_broker(portfolio)
        params = broker._sign_request({"symbol": "BTCUSDT"})
        assert "timestamp" in params
        assert "signature" in params
        # Signature should be a hex string (64 chars for SHA256)
        assert len(params["signature"]) == 64

    def test_get_headers_contains_api_key(self, portfolio):
        """Headers must include the API key for authentication."""
        broker = self._make_broker(portfolio)
        headers = broker._get_headers()
        assert headers["X-MBX-APIKEY"] == "test_api_key"

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance_success(self, mock_get, portfolio):
        """Should parse Binance balance response correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "balances": [
                {"asset": "USDT", "free": "10000.00", "locked": "500.00"},
                {"asset": "BTC", "free": "0.5", "locked": "0.0"},
                {"asset": "ETH", "free": "0.0", "locked": "0.0"},  # Zero balance, should be excluded
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker(portfolio)
        balances = broker.get_account_balance()

        assert "USDT" in balances
        assert balances["USDT"]["free"] == 10000.0
        assert balances["USDT"]["locked"] == 500.0
        assert balances["USDT"]["total"] == 10500.0
        assert "BTC" in balances
        assert "ETH" not in balances  # Zero balance filtered out

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance_failure(self, mock_get, portfolio):
        """Should return empty dict on API failure."""
        mock_get.side_effect = Exception("Connection timeout")
        broker = self._make_broker(portfolio)
        balances = broker.get_account_balance()
        assert balances == {}

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_success(self, mock_get, portfolio):
        """Should parse price ticker response correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"price": "67500.25"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker(portfolio)
        price = broker.get_price("BTCUSDT")
        assert price == 67500.25

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_failure(self, mock_get, portfolio):
        """Should return 0.0 on price fetch failure."""
        mock_get.side_effect = Exception("Network error")
        broker = self._make_broker(portfolio)
        price = broker.get_price("BTCUSDT")
        assert price == 0.0

    @patch("src.execution.live_broker.requests.post")
    def test_submit_market_order_success(self, mock_post, portfolio):
        """Market order should be submitted and filled correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderId": 123456,
            "status": "FILLED",
            "price": "67000.00",
            "executedQty": "0.01",
            "cummulativeQuoteQty": "670.00",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = self._make_broker(portfolio)
        order = _make_order()
        result = broker.submit_order(order)

        assert result.exchange_order_id == "123456"
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 0.01
        assert len(broker.filled_orders) == 1
        assert len(broker.all_orders) == 1

    @patch("src.execution.live_broker.requests.post")
    def test_submit_order_rejected(self, mock_post, portfolio):
        """Should handle Binance order rejection gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"msg": "Insufficient balance"}
        mock_response.raise_for_status.side_effect = Exception("HTTP 400")
        mock_post.return_value = mock_response

        broker = self._make_broker(portfolio)
        order = _make_order()
        result = broker.submit_order(order)

        assert result.status == OrderStatus.REJECTED
        assert len(broker.filled_orders) == 0

    @patch("src.execution.live_broker.requests.post")
    def test_submit_limit_order(self, mock_post, portfolio):
        """Limit order should include price and timeInForce parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderId": 789,
            "status": "NEW",
            "price": "60000.00",
            "executedQty": "0",
            "cummulativeQuoteQty": "0",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = self._make_broker(portfolio)
        order = _make_order(order_type=OrderType.LIMIT, price=60000.0)
        result = broker.submit_order(order)

        assert result.exchange_order_id == "789"
        assert result.status == OrderStatus.SUBMITTED
        # Verify the API was called with price parameter
        call_kwargs = mock_post.call_args
        assert "60000.00" in str(call_kwargs)

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_success(self, mock_delete, portfolio):
        """Should cancel order and return True on success."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_delete.return_value = mock_response

        broker = self._make_broker(portfolio)
        result = broker.cancel_order("123456", "BTCUSDT")
        assert result is True

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_failure(self, mock_delete, portfolio):
        """Should return False on cancel failure."""
        mock_delete.side_effect = Exception("Not found")
        broker = self._make_broker(portfolio)
        result = broker.cancel_order("999", "BTCUSDT")
        assert result is False

    def test_execution_stats(self, portfolio):
        """Execution stats should reflect correct counts."""
        broker = self._make_broker(portfolio)
        stats = broker.get_execution_stats()
        assert stats["broker"] == "Binance"
        assert stats["mode"] == "LIVE"
        assert stats["total_orders"] == 0
        assert stats["filled_orders"] == 0
        assert stats["fill_rate"] == 0.0

    def test_map_order_type(self, portfolio):
        """Internal order types should map to Binance API strings."""
        broker = self._make_broker(portfolio)
        assert broker._map_order_type(OrderType.MARKET) == "MARKET"
        assert broker._map_order_type(OrderType.LIMIT) == "LIMIT"
        assert broker._map_order_type(OrderType.STOP_LOSS) == "STOP_LOSS_LIMIT"

    def test_map_status(self, portfolio):
        """Binance status strings should map to internal OrderStatus."""
        broker = self._make_broker(portfolio)
        assert broker._map_status("NEW") == OrderStatus.SUBMITTED
        assert broker._map_status("FILLED") == OrderStatus.FILLED
        assert broker._map_status("CANCELED") == OrderStatus.CANCELLED
        assert broker._map_status("REJECTED") == OrderStatus.REJECTED


# ======================================================================
# WazirX Live Broker Tests
# ======================================================================

class TestWazirXLiveBroker:
    """Tests for WazirXLiveBroker with mocked WazirX API responses."""

    def _make_broker(self, portfolio):
        """Create a WazirX broker instance for testing."""
        return WazirXLiveBroker(
            api_key="test_wazirx_key",
            api_secret="test_wazirx_secret",
            portfolio=portfolio,
        )

    def test_init(self, portfolio):
        """WazirX broker should initialize with correct base URL."""
        broker = self._make_broker(portfolio)
        assert broker.BASE_URL == "https://api.wazirx.com"

    def test_sign_request_adds_fields(self, portfolio):
        """Signed request should contain timestamp, recvWindow, and signature."""
        broker = self._make_broker(portfolio)
        params = broker._sign_request({"symbol": "btcinr"})
        assert "timestamp" in params
        assert "recvWindow" in params
        assert "signature" in params
        assert len(params["signature"]) == 64

    def test_get_headers(self, portfolio):
        """Headers should include X-Api-Key for WazirX authentication."""
        broker = self._make_broker(portfolio)
        headers = broker._get_headers()
        assert headers["X-Api-Key"] == "test_wazirx_key"

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance_success(self, mock_get, portfolio):
        """Should parse WazirX fund balances correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"asset": "inr", "free": "100000", "locked": "5000"},
            {"asset": "btc", "free": "0.1", "locked": "0"},
            {"asset": "usdt", "free": "0", "locked": "0"},  # Zero — should be excluded
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker(portfolio)
        balances = broker.get_account_balance()

        assert "inr" in balances
        assert balances["inr"]["total"] == 105000.0
        assert "btc" in balances
        assert "usdt" not in balances

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance_failure(self, mock_get, portfolio):
        """Should return empty dict on API error."""
        mock_get.side_effect = Exception("Timeout")
        broker = self._make_broker(portfolio)
        assert broker.get_account_balance() == {}

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_success(self, mock_get, portfolio):
        """Should return last traded price from WazirX ticker."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"lastPrice": "5600000"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker(portfolio)
        price = broker.get_price("btcinr")
        assert price == 5600000.0

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_failure(self, mock_get, portfolio):
        """Should return 0.0 on price fetch failure."""
        mock_get.side_effect = Exception("Network error")
        broker = self._make_broker(portfolio)
        assert broker.get_price("btcinr") == 0.0

    @patch("src.execution.live_broker.requests.post")
    def test_submit_market_order_success(self, mock_post, portfolio):
        """Market order on WazirX should be submitted correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderId": 7001,
            "status": "done",
            "price": "5600000",
            "executedQty": "0.01",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = self._make_broker(portfolio)
        order = _make_order(quantity=0.01)
        result = broker.submit_order(order)

        assert result.exchange_order_id == "7001"
        assert result.status == OrderStatus.FILLED
        assert len(broker.filled_orders) == 1

    @patch("src.execution.live_broker.requests.post")
    def test_submit_order_network_error(self, mock_post, portfolio):
        """Should reject order on network failure without crashing."""
        mock_post.side_effect = Exception("Connection refused")
        broker = self._make_broker(portfolio)
        order = _make_order()
        result = broker.submit_order(order)

        assert result.status == OrderStatus.REJECTED
        assert "WazirX error" in result.notes

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_success(self, mock_delete, portfolio):
        """Should cancel WazirX order successfully."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_delete.return_value = mock_response

        broker = self._make_broker(portfolio)
        assert broker.cancel_order("7001") is True

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_failure(self, mock_delete, portfolio):
        """Should return False on cancel failure."""
        mock_delete.side_effect = Exception("Not found")
        broker = self._make_broker(portfolio)
        assert broker.cancel_order("999") is False

    def test_execution_stats(self, portfolio):
        """Stats should report WazirX broker and INR currency."""
        broker = self._make_broker(portfolio)
        stats = broker.get_execution_stats()
        assert stats["broker"] == "WazirX"
        assert stats["currency"] == "INR"
        assert stats["total_orders"] == 0


# ======================================================================
# Delta Exchange Live Broker Tests
# ======================================================================

class TestDeltaExchangeLiveBroker:
    """Tests for DeltaExchangeLiveBroker with mocked Delta API responses."""

    def _make_broker(self, portfolio, testnet=True, currency="INR"):
        """Create a Delta Exchange broker instance for testing."""
        return DeltaExchangeLiveBroker(
            api_key="test_delta_key",
            api_secret="test_delta_secret",
            portfolio=portfolio,
            testnet=testnet,
            currency=currency,
        )

    def test_init_testnet(self, portfolio):
        """Testnet mode should use Delta testnet URL."""
        broker = self._make_broker(portfolio, testnet=True)
        assert broker.base_url == "https://testnet-api.delta.exchange"

    def test_init_live(self, portfolio):
        """Live mode should use production Delta URL."""
        broker = self._make_broker(portfolio, testnet=False)
        assert broker.base_url == "https://api.india.delta.exchange"

    def test_init_currency(self, portfolio):
        """Currency should be stored and uppercased."""
        broker_inr = self._make_broker(portfolio, currency="inr")
        assert broker_inr.currency == "INR"
        broker_usd = self._make_broker(portfolio, currency="USD")
        assert broker_usd.currency == "USD"

    def test_generate_signature(self, portfolio):
        """Signature should be a valid HMAC-SHA256 hex string."""
        broker = self._make_broker(portfolio)
        headers = broker._generate_signature("GET", "/v2/wallet/balances")
        assert "api-key" in headers
        assert headers["api-key"] == "test_delta_key"
        assert "signature" in headers
        assert len(headers["signature"]) == 64
        assert "timestamp" in headers

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance_success(self, mock_get, portfolio):
        """Should parse Delta Exchange wallet response correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [
                {"asset_symbol": "INR", "balance": "100000", "available_balance": "95000"},
                {"asset_symbol": "BTC", "balance": "0.5", "available_balance": "0.5"},
                {"asset_symbol": "USDT", "balance": "0", "available_balance": "0"},  # Zero — excluded
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker(portfolio)
        balances = broker.get_account_balance()

        assert "INR" in balances
        assert balances["INR"]["total"] == 100000.0
        assert balances["INR"]["available"] == 95000.0
        assert balances["INR"]["reserved"] == 5000.0
        assert "BTC" in balances
        assert "USDT" not in balances

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance_failure(self, mock_get, portfolio):
        """Should return empty dict on API failure."""
        mock_get.side_effect = Exception("Timeout")
        broker = self._make_broker(portfolio)
        assert broker.get_account_balance() == {}

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_success(self, mock_get, portfolio):
        """Should return mark price from Delta Exchange ticker."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"mark_price": "67500.50", "close": "67490.00"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker(portfolio)
        price = broker.get_price("BTCUSDT")
        assert price == 67500.50

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_fallback_to_close(self, mock_get, portfolio):
        """Should fall back to close price if mark_price is missing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"mark_price": "0", "close": "67490.00"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker(portfolio)
        price = broker.get_price("BTCUSDT")
        assert price == 67490.0

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_failure(self, mock_get, portfolio):
        """Should return 0.0 on price fetch failure."""
        mock_get.side_effect = Exception("Network error")
        broker = self._make_broker(portfolio)
        assert broker.get_price("BTCUSDT") == 0.0

    @patch("src.execution.live_broker.requests.post")
    def test_submit_market_order_success(self, mock_post, portfolio):
        """Market order on Delta should be submitted and filled."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "id": 55001,
                "state": "closed",
                "average_fill_price": "67000",
                "size": "0.01",
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = self._make_broker(portfolio)
        order = _make_order(quantity=0.01)
        result = broker.submit_order(order)

        assert result.exchange_order_id == "55001"
        assert result.status == OrderStatus.FILLED
        assert result.filled_price == 67000.0
        assert len(broker.filled_orders) == 1

    @patch("src.execution.live_broker.requests.post")
    def test_submit_order_pending(self, mock_post, portfolio):
        """Open/pending orders should be tracked as SUBMITTED."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "id": 55002,
                "state": "open",
                "average_fill_price": "0",
                "size": "0.01",
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = self._make_broker(portfolio)
        order = _make_order()
        result = broker.submit_order(order)

        assert result.status == OrderStatus.SUBMITTED
        assert len(broker.filled_orders) == 0

    @patch("src.execution.live_broker.requests.post")
    def test_submit_order_network_error(self, mock_post, portfolio):
        """Should reject order on network failure."""
        mock_post.side_effect = Exception("Connection refused")
        broker = self._make_broker(portfolio)
        order = _make_order()
        result = broker.submit_order(order)

        assert result.status == OrderStatus.REJECTED
        assert "Delta error" in result.notes

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_success(self, mock_delete, portfolio):
        """Should cancel Delta order and return True."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_delete.return_value = mock_response

        broker = self._make_broker(portfolio)
        assert broker.cancel_order("55001") is True

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_failure(self, mock_delete, portfolio):
        """Should return False on cancel failure."""
        mock_delete.side_effect = Exception("Not found")
        broker = self._make_broker(portfolio)
        assert broker.cancel_order("999") is False

    def test_execution_stats_inr(self, portfolio):
        """Stats should reflect Delta Exchange broker and INR currency."""
        broker = self._make_broker(portfolio, currency="INR")
        stats = broker.get_execution_stats()
        assert stats["broker"] == "Delta Exchange"
        assert stats["currency"] == "INR"

    def test_execution_stats_usdt(self, portfolio):
        """Stats should reflect USDT currency when configured."""
        broker = self._make_broker(portfolio, currency="USD")
        stats = broker.get_execution_stats()
        assert stats["currency"] == "USD"

    def test_map_order_type(self, portfolio):
        """Internal order types should map to Delta Exchange API types."""
        broker = self._make_broker(portfolio)
        assert broker._map_order_type(OrderType.MARKET) == "market_order"
        assert broker._map_order_type(OrderType.LIMIT) == "limit_order"
        assert broker._map_order_type(OrderType.STOP_LOSS) == "stop_market_order"

    def test_map_status(self, portfolio):
        """Delta status strings should map to internal OrderStatus."""
        broker = self._make_broker(portfolio)
        assert broker._map_status("open") == OrderStatus.SUBMITTED
        assert broker._map_status("closed") == OrderStatus.FILLED
        assert broker._map_status("cancelled") == OrderStatus.CANCELLED
        assert broker._map_status("rejected") == OrderStatus.REJECTED

    def test_product_symbol_mapping_inr(self, portfolio):
        """INR broker should use BTCINR product symbol."""
        broker = self._make_broker(portfolio, currency="INR")
        assert broker.currency == "INR"
        # Verify the product symbols dict has expected mappings
        assert "BTC/USDT" in broker.PRODUCT_SYMBOLS
        assert "BTC/INR" in broker.PRODUCT_SYMBOLS


# ──────────────────────────────────────────────
# XM Broker Tests (MetaTrader 5 Web API)
# ──────────────────────────────────────────────

class TestXMBroker:
    """Tests for XM broker via MetaTrader 5 Web API."""

    @staticmethod
    def _make_broker(portfolio, demo=True):
        """Create an XM broker with test credentials."""
        return XMBroker(
            mt5_login="12345678",
            mt5_password="test_password",
            mt5_server="XMGlobal-MT5",
            portfolio=portfolio,
            demo=demo,
        )

    def test_initialization_demo(self, portfolio):
        """XM broker should initialize in demo mode by default."""
        broker = self._make_broker(portfolio, demo=True)
        assert broker.demo is True
        assert broker.base_url == XMBroker.DEMO_BASE_URL
        assert broker.mt5_login == "12345678"
        assert broker.mt5_server == "XMGlobal-MT5"

    def test_initialization_live(self, portfolio):
        """XM broker should use live URL when demo=False."""
        broker = self._make_broker(portfolio, demo=False)
        assert broker.demo is False
        assert broker.base_url == XMBroker.LIVE_BASE_URL

    def test_symbol_mapping_gold(self, portfolio):
        """XAU/USD should map to XAUUSD for XM MT5."""
        broker = self._make_broker(portfolio)
        assert broker._map_symbol("XAU/USD") == "XAUUSD"

    def test_symbol_mapping_btc(self, portfolio):
        """BTC/USD and BTC/USDT should both map to BTCUSD on XM."""
        broker = self._make_broker(portfolio)
        assert broker._map_symbol("BTC/USD") == "BTCUSD"
        assert broker._map_symbol("BTC/USDT") == "BTCUSD"

    @patch("src.execution.live_broker.requests.post")
    def test_authentication_success(self, mock_post, portfolio):
        """Successful MT5 login should store session token."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"token": "test_session_token_123"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        broker = self._make_broker(portfolio)
        result = broker._authenticate()
        assert result is True
        assert broker._session_token == "test_session_token_123"

    @patch("src.execution.live_broker.requests.post")
    def test_authentication_failure(self, mock_post, portfolio):
        """Failed MT5 login should return False and not set token."""
        mock_post.side_effect = Exception("Connection refused")
        broker = self._make_broker(portfolio)
        result = broker._authenticate()
        assert result is False
        assert broker._session_token is None

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance(self, mock_get, portfolio):
        """Should parse MT5 account balance response correctly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "balance": 50000.0,
            "equity": 49500.0,
            "margin": 2000.0,
            "free_margin": 47500.0,
            "margin_level": 2475.0,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        balances = broker.get_account_balance()
        assert "USD" in balances
        assert balances["USD"]["balance"] == 50000.0
        assert balances["USD"]["free_margin"] == 47500.0

    @patch("src.execution.live_broker.requests.get")
    def test_get_account_balance_error(self, mock_get, portfolio):
        """Balance fetch failure should return empty dict."""
        mock_get.side_effect = Exception("Network error")
        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        balances = broker.get_account_balance()
        assert balances == {}

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_gold(self, mock_get, portfolio):
        """Should return mid price (avg of bid and ask) for Gold."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"bid": 2340.50, "ask": 2341.00}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        price = broker.get_price("XAU/USD")
        assert price == pytest.approx(2340.75, rel=1e-4)

    @patch("src.execution.live_broker.requests.get")
    def test_get_price_error(self, mock_get, portfolio):
        """Price fetch failure should return 0.0."""
        mock_get.side_effect = Exception("Timeout")
        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        price = broker.get_price("XAUUSD")
        assert price == 0.0

    @patch("src.execution.live_broker.requests.post")
    def test_submit_order_success(self, mock_post, portfolio):
        """Successful order should update order with exchange data."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "order_id": "987654",
            "status": "filled",
            "price": 2340.50,
            "volume": 0.1,
            "commission": 0.35,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        order = Order(
            symbol="XAU/USD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=0.1, price=2340.0,
        )
        result = broker.submit_order(order)
        assert result.exchange_order_id == "987654"
        assert result.status == OrderStatus.FILLED
        assert len(broker.filled_orders) == 1

    @patch("src.execution.live_broker.requests.post")
    def test_submit_order_rejected(self, mock_post, portfolio):
        """HTTP error should result in rejected order status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"message": "Insufficient margin"}
        mock_resp.raise_for_status.side_effect = __import__("requests").exceptions.HTTPError(response=mock_resp)
        mock_post.return_value = mock_resp

        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        order = Order(
            symbol="BTC/USD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=0.01, price=107000.0,
        )
        result = broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_success(self, mock_delete, portfolio):
        """Successful cancellation should return True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_delete.return_value = mock_resp

        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        result = broker.cancel_order("987654")
        assert result is True

    @patch("src.execution.live_broker.requests.delete")
    def test_cancel_order_failure(self, mock_delete, portfolio):
        """Failed cancellation should return False."""
        mock_delete.side_effect = Exception("Order not found")
        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        result = broker.cancel_order("999999")
        assert result is False

    @patch("src.execution.live_broker.requests.get")
    def test_get_open_positions(self, mock_get, portfolio):
        """Should parse open positions list from MT5 API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "ticket": 12345,
                "symbol": "XAUUSD",
                "volume": 0.1,
                "type": "BUY",
                "price_open": 2340.0,
                "price_current": 2350.0,
                "profit": 100.0,
                "sl": 2320.0,
                "tp": 2380.0,
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        positions = broker.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "XAUUSD"
        assert positions[0]["profit"] == 100.0

    @patch("src.execution.live_broker.requests.post")
    def test_close_position_success(self, mock_post, portfolio):
        """Successful position close should return True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        broker = self._make_broker(portfolio)
        broker._session_token = "test_token"
        result = broker.close_position(12345)
        assert result is True

    def test_execution_stats_demo(self, portfolio):
        """Stats should reflect XM broker in demo mode."""
        broker = self._make_broker(portfolio, demo=True)
        stats = broker.get_execution_stats()
        assert stats["broker"] == "XM (MT5)"
        assert stats["mode"] == "DEMO"
        assert stats["server"] == "XMGlobal-MT5"

    def test_execution_stats_live(self, portfolio):
        """Stats should reflect live mode when demo=False."""
        broker = self._make_broker(portfolio, demo=False)
        stats = broker.get_execution_stats()
        assert stats["mode"] == "LIVE"

    def test_map_order_type(self, portfolio):
        """Internal order types should map to MT5 order type strings."""
        broker = self._make_broker(portfolio)
        assert broker._map_order_type(OrderType.MARKET) == "ORDER_TYPE_BUY"
        assert broker._map_order_type(OrderType.LIMIT) == "ORDER_TYPE_BUY_LIMIT"
        assert broker._map_order_type(OrderType.STOP_LOSS) == "ORDER_TYPE_BUY_STOP"

    def test_map_status(self, portfolio):
        """XM MT5 status strings should map to internal OrderStatus."""
        broker = self._make_broker(portfolio)
        assert broker._map_status("filled") == OrderStatus.FILLED
        assert broker._map_status("placed") == OrderStatus.SUBMITTED
        assert broker._map_status("cancelled") == OrderStatus.CANCELLED
        assert broker._map_status("rejected") == OrderStatus.REJECTED
        assert broker._map_status("ORDER_STATE_FILLED") == OrderStatus.FILLED
        assert broker._map_status("ORDER_STATE_CANCELED") == OrderStatus.CANCELLED
