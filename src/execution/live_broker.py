"""
Live broker implementations for real-money trading.
Supports Binance (USDT), WazirX (INR), Delta Exchange (INR + USDT),
and XM (Gold XAU/USD + BTC via MetaTrader 5 REST API).

All brokers use HMAC-SHA256 signed API requests for authentication.
API keys must be configured via environment variables or settings.

Security: API keys are never logged or stored in code.
"""

import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from loguru import logger

from config.settings import OrderSide, OrderStatus, OrderType
from src.execution.order import Order
from src.execution.portfolio import Portfolio


class BinanceLiveBroker:
    """
    Live broker for Binance exchange (USDT-based trading).

    Connects to Binance API to execute real BTC/USDT trades.
    Requires BINANCE_API_KEY and BINANCE_API_SECRET environment variables.

    Supported operations:
        - Market and limit order execution
        - Account balance retrieval
        - Open order management
        - Order status checking
    """

    # Binance API endpoints
    BASE_URL = "https://api.binance.com"
    ORDER_ENDPOINT = "/api/v3/order"
    ACCOUNT_ENDPOINT = "/api/v3/account"
    TICKER_ENDPOINT = "/api/v3/ticker/price"

    def __init__(self, api_key: str, api_secret: str, portfolio: Portfolio, testnet: bool = False):
        """
        Initialize Binance live broker.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            portfolio: Portfolio instance to track positions
            testnet: If True, use Binance testnet (recommended for testing)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.portfolio = portfolio

        # Use testnet URL if specified (for safe testing)
        if testnet:
            self.BASE_URL = "https://testnet.binance.vision"
            logger.info("Binance broker initialized in TESTNET mode")
        else:
            logger.info("Binance broker initialized in LIVE mode")

        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.all_orders: List[Order] = []

    def _sign_request(self, params: dict) -> dict:
        """
        Sign API request parameters with HMAC-SHA256.

        Args:
            params: Request parameters to sign

        Returns:
            Parameters dict with signature appended
        """
        params["timestamp"] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get_headers(self) -> dict:
        """Return authenticated request headers."""
        return {"X-MBX-APIKEY": self.api_key}

    def get_account_balance(self) -> dict:
        """
        Fetch account balances from Binance.

        Returns:
            Dict with asset balances (e.g., {"USDT": 10000.0, "BTC": 0.5})
        """
        params = self._sign_request({})
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.ACCOUNT_ENDPOINT}",
                params=params,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            # Filter for non-zero balances
            balances = {}
            for asset in data.get("balances", []):
                free = float(asset["free"])
                locked = float(asset["locked"])
                if free > 0 or locked > 0:
                    balances[asset["asset"]] = {
                        "free": free,
                        "locked": locked,
                        "total": free + locked,
                    }
            return balances
        except Exception as e:
            logger.error(f"Failed to fetch Binance account: {e}")
            return {}

    def get_price(self, symbol: str = "BTCUSDT") -> float:
        """
        Fetch current market price for a symbol.

        Args:
            symbol: Binance symbol (e.g., "BTCUSDT")

        Returns:
            Current price as float
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.TICKER_ENDPOINT}",
                params={"symbol": symbol},
                timeout=10,
            )
            response.raise_for_status()
            return float(response.json()["price"])
        except Exception as e:
            logger.error(f"Failed to fetch Binance price: {e}")
            return 0.0

    def submit_order(self, order: Order) -> Order:
        """
        Submit a real order to Binance.

        Args:
            order: Order object to execute

        Returns:
            Updated order with exchange order ID and status
        """
        # Map internal symbol to Binance symbol format
        binance_symbol = order.symbol.replace("/", "").replace("BTC/USDT", "BTCUSDT")

        # Build order parameters
        params = {
            "symbol": binance_symbol,
            "side": "BUY" if order.side == OrderSide.BUY else "SELL",
            "type": self._map_order_type(order.order_type),
            "quantity": f"{order.quantity:.6f}",
        }

        # Add price for limit orders
        if order.order_type == OrderType.LIMIT:
            params["price"] = f"{order.price:.2f}"
            params["timeInForce"] = "GTC"

        # Add stop price for stop orders
        if order.order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
            params["stopPrice"] = f"{order.stop_price:.2f}"

        # Sign and send the request
        params = self._sign_request(params)
        try:
            response = requests.post(
                f"{self.BASE_URL}{self.ORDER_ENDPOINT}",
                params=params,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            # Update order with exchange data
            order.exchange_order_id = str(result.get("orderId", ""))
            order.status = self._map_status(result.get("status", ""))
            order.filled_price = float(result.get("price", 0)) or order.price
            order.filled_quantity = float(result.get("executedQty", 0))
            order.commission = float(result.get("cummulativeQuoteQty", 0)) * 0.001  # 0.1% fee estimate

            if order.status == OrderStatus.FILLED:
                self.filled_orders.append(order)
                # Update portfolio with the fill
                self._process_fill(order)

            logger.info(
                f"Binance order placed: {order.exchange_order_id} "
                f"{order.side.value.upper()} {order.quantity:.6f} {order.symbol} "
                f"Status: {order.status.value}"
            )
        except requests.exceptions.HTTPError as e:
            error_msg = e.response.json().get("msg", str(e)) if e.response else str(e)
            order.reject(f"Binance rejected: {error_msg}")
            logger.error(f"Binance order rejected: {error_msg}")
        except Exception as e:
            order.reject(f"Binance error: {str(e)}")
            logger.error(f"Binance order failed: {e}")

        self.all_orders.append(order)
        return order

    def _process_fill(self, order: Order):
        """Update portfolio after an order is filled on the exchange."""
        side = "buy" if order.side == OrderSide.BUY else "sell"
        self.portfolio.open_position(
            symbol=order.symbol,
            side=side,
            quantity=order.filled_quantity,
            price=order.filled_price,
            commission=order.commission,
        )

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to Binance API order type string."""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT_LIMIT",
        }
        return mapping.get(order_type, "MARKET")

    def _map_status(self, binance_status: str) -> OrderStatus:
        """Map Binance order status to internal OrderStatus."""
        mapping = {
            "NEW": OrderStatus.SUBMITTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(binance_status, OrderStatus.PENDING)

    def cancel_order(self, order_id: str, symbol: str = "BTCUSDT") -> bool:
        """
        Cancel an open order on Binance.

        Args:
            order_id: Exchange order ID to cancel
            symbol: Binance trading symbol

        Returns:
            True if cancellation was successful
        """
        params = self._sign_request({
            "symbol": symbol,
            "orderId": order_id,
        })
        try:
            response = requests.delete(
                f"{self.BASE_URL}{self.ORDER_ENDPOINT}",
                params=params,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"Binance order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel Binance order {order_id}: {e}")
            return False

    def get_execution_stats(self) -> dict:
        """Return broker execution statistics."""
        return {
            "broker": "Binance",
            "mode": "LIVE",
            "total_orders": len(self.all_orders),
            "filled_orders": len(self.filled_orders),
            "fill_rate": len(self.filled_orders) / max(1, len(self.all_orders)) * 100,
        }


class WazirXLiveBroker:
    """
    Live broker for WazirX exchange (INR-based trading).

    Connects to WazirX API to execute BTC/INR trades.
    Requires WAZIRX_API_KEY and WAZIRX_API_SECRET environment variables.

    WazirX is India's largest crypto exchange, supporting INR deposits/withdrawals.
    """

    # WazirX API endpoints
    BASE_URL = "https://api.wazirx.com"
    ORDER_ENDPOINT = "/sapi/v1/order"
    ACCOUNT_ENDPOINT = "/sapi/v1/funds"
    TICKER_ENDPOINT = "/sapi/v1/ticker/24hr"

    def __init__(self, api_key: str, api_secret: str, portfolio: Portfolio):
        """
        Initialize WazirX live broker.

        Args:
            api_key: WazirX API key
            api_secret: WazirX API secret
            portfolio: Portfolio instance to track positions
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.portfolio = portfolio

        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.all_orders: List[Order] = []

        logger.info("WazirX broker initialized for INR trading")

    def _sign_request(self, params: dict) -> dict:
        """
        Sign API request with HMAC-SHA256 for WazirX authentication.

        Args:
            params: Request parameters to sign

        Returns:
            Parameters with signature appended
        """
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 10000
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get_headers(self) -> dict:
        """Return authenticated request headers for WazirX."""
        return {"X-Api-Key": self.api_key}

    def get_account_balance(self) -> dict:
        """
        Fetch account balances from WazirX.

        Returns:
            Dict with asset balances (e.g., {"inr": 100000.0, "btc": 0.5})
        """
        params = self._sign_request({})
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.ACCOUNT_ENDPOINT}",
                params=params,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            balances = {}
            for asset in data:
                free = float(asset.get("free", 0))
                locked = float(asset.get("locked", 0))
                if free > 0 or locked > 0:
                    balances[asset["asset"]] = {
                        "free": free,
                        "locked": locked,
                        "total": free + locked,
                    }
            return balances
        except Exception as e:
            logger.error(f"Failed to fetch WazirX account: {e}")
            return {}

    def get_price(self, symbol: str = "btcinr") -> float:
        """
        Fetch current market price for a symbol on WazirX.

        Args:
            symbol: WazirX symbol (e.g., "btcinr")

        Returns:
            Current price as float
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.TICKER_ENDPOINT}",
                params={"symbol": symbol},
                timeout=10,
            )
            response.raise_for_status()
            return float(response.json().get("lastPrice", 0))
        except Exception as e:
            logger.error(f"Failed to fetch WazirX price: {e}")
            return 0.0

    def submit_order(self, order: Order) -> Order:
        """
        Submit a real order to WazirX exchange.

        Args:
            order: Order object to execute

        Returns:
            Updated order with exchange response data
        """
        # Map symbol: BTC/USDT -> btcinr for WazirX
        wazirx_symbol = "btcinr"

        # Build order parameters
        params = {
            "symbol": wazirx_symbol,
            "side": "buy" if order.side == OrderSide.BUY else "sell",
            "type": "limit" if order.order_type == OrderType.LIMIT else "market",
            "quantity": f"{order.quantity:.6f}",
        }

        # Add price for limit orders
        if order.order_type == OrderType.LIMIT:
            params["price"] = f"{order.price:.2f}"

        # Sign and send
        params = self._sign_request(params)
        try:
            response = requests.post(
                f"{self.BASE_URL}{self.ORDER_ENDPOINT}",
                params=params,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            order.exchange_order_id = str(result.get("orderId", ""))
            order.status = OrderStatus.FILLED if result.get("status") == "done" else OrderStatus.SUBMITTED
            order.filled_price = float(result.get("price", 0)) or order.price
            order.filled_quantity = float(result.get("executedQty", 0)) or order.quantity
            order.commission = order.filled_quantity * order.filled_price * 0.002  # 0.2% WazirX fee

            if order.status == OrderStatus.FILLED:
                self.filled_orders.append(order)
                self._process_fill(order)

            logger.info(
                f"WazirX order placed: {order.exchange_order_id} "
                f"{order.side.value.upper()} {order.quantity:.6f} BTC/INR "
                f"Status: {order.status.value}"
            )
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_msg = e.response.json().get("message", str(e))
            except Exception:
                pass
            order.reject(f"WazirX rejected: {error_msg}")
            logger.error(f"WazirX order rejected: {error_msg}")
        except Exception as e:
            order.reject(f"WazirX error: {str(e)}")
            logger.error(f"WazirX order failed: {e}")

        self.all_orders.append(order)
        return order

    def _process_fill(self, order: Order):
        """Update portfolio after a WazirX order fill."""
        side = "buy" if order.side == OrderSide.BUY else "sell"
        self.portfolio.open_position(
            symbol="BTC/INR",
            side=side,
            quantity=order.filled_quantity,
            price=order.filled_price,
            commission=order.commission,
        )

    def cancel_order(self, order_id: str, symbol: str = "btcinr") -> bool:
        """
        Cancel an open order on WazirX.

        Args:
            order_id: Exchange order ID
            symbol: WazirX symbol

        Returns:
            True if cancellation succeeded
        """
        params = self._sign_request({
            "symbol": symbol,
            "orderId": order_id,
        })
        try:
            response = requests.delete(
                f"{self.BASE_URL}{self.ORDER_ENDPOINT}",
                params=params,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"WazirX order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel WazirX order {order_id}: {e}")
            return False

    def get_execution_stats(self) -> dict:
        """Return broker execution statistics."""
        return {
            "broker": "WazirX",
            "mode": "LIVE",
            "currency": "INR",
            "total_orders": len(self.all_orders),
            "filled_orders": len(self.filled_orders),
            "fill_rate": len(self.filled_orders) / max(1, len(self.all_orders)) * 100,
        }


class DeltaExchangeLiveBroker:
    """
    Live broker for Delta Exchange (Indian crypto exchange).

    Delta Exchange supports both INR and USDT markets for BTC trading,
    including spot and derivatives (futures/options).
    API docs: https://docs.delta.exchange

    Requires DELTA_API_KEY and DELTA_API_SECRET environment variables.
    Supports testnet mode for safe testing before live trading.
    """

    # Delta Exchange API endpoints — India endpoint for Indian accounts
    BASE_URL = "https://api.india.delta.exchange"
    BASE_URL_GLOBAL = "https://api.delta.exchange"
    TESTNET_URL = "https://testnet-api.delta.exchange"
    ORDER_ENDPOINT = "/v2/orders"
    POSITION_ENDPOINT = "/v2/positions"
    WALLET_ENDPOINT = "/v2/wallet/balances"
    TICKER_ENDPOINT = "/v2/tickers"

    # Delta product IDs for BTC (may vary — fetched dynamically in production)
    PRODUCT_SYMBOLS = {
        "BTC/USDT": "BTCUSDT",
        "BTC/INR": "BTCINR",
    }

    def __init__(self, api_key: str, api_secret: str, portfolio: Portfolio,
                 testnet: bool = False, currency: str = "INR"):
        """
        Initialize Delta Exchange broker.

        Args:
            api_key: Delta Exchange API key
            api_secret: Delta Exchange API secret
            portfolio: Portfolio instance for position tracking
            testnet: If True, use Delta testnet for safe testing
            currency: Trading currency — "INR" or "USDT"
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.portfolio = portfolio
        self.currency = currency.upper()

        # Select base URL based on testnet flag
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL

        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.all_orders: List[Order] = []

        mode_label = "TESTNET" if testnet else "LIVE"
        logger.info(f"Delta Exchange broker initialized ({mode_label}, {self.currency})")

    def _generate_signature(self, method: str, endpoint: str, payload: str = "") -> dict:
        """
        Generate HMAC-SHA256 signature for Delta Exchange API authentication.

        Delta uses a different signing scheme than Binance/WazirX:
        signature = HMAC-SHA256(secret, method + timestamp + endpoint + payload)

        Args:
            method: HTTP method ("GET" or "POST")
            endpoint: API endpoint path
            payload: JSON body string for POST requests

        Returns:
            Dict of authentication headers
        """
        timestamp = str(int(time.time()))
        # Delta signature format: method + timestamp + path + body
        signature_data = method + timestamp + endpoint + payload
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "api-key": self.api_key,
            "signature": signature,
            "timestamp": timestamp,
            "Content-Type": "application/json",
        }

    def get_account_balance(self) -> dict:
        """
        Fetch wallet balances from Delta Exchange.

        Returns:
            Dict with asset balances (e.g., {"INR": 100000.0, "BTC": 0.5})
        """
        endpoint = self.WALLET_ENDPOINT
        headers = self._generate_signature("GET", endpoint)

        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            balances = {}
            for asset in data.get("result", []):
                balance = float(asset.get("balance", 0))
                available = float(asset.get("available_balance", 0))
                if balance > 0:
                    balances[asset.get("asset_symbol", "unknown")] = {
                        "total": balance,
                        "available": available,
                        "reserved": balance - available,
                    }
            return balances
        except Exception as e:
            logger.error(f"Failed to fetch Delta Exchange balance: {e}")
            return {}

    def get_price(self, symbol: str = "BTCUSDT") -> float:
        """
        Fetch current market price from Delta Exchange.

        Args:
            symbol: Delta product symbol (e.g., "BTCUSDT" or "BTCINR")

        Returns:
            Current mark price as float
        """
        endpoint = f"{self.TICKER_ENDPOINT}/{symbol}"
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            result = data.get("result", {})
            # Delta returns mark_price for derivatives, close for spot
            # Convert to float first to handle string "0" correctly
            mark = float(result.get("mark_price", 0) or 0)
            close = float(result.get("close", 0) or 0)
            return mark if mark > 0 else close
        except Exception as e:
            logger.error(f"Failed to fetch Delta Exchange price: {e}")
            return 0.0

    def submit_order(self, order: Order) -> Order:
        """
        Submit a real order to Delta Exchange.

        Args:
            order: Order object to execute

        Returns:
            Updated order with exchange response data
        """
        # Map internal symbol to Delta product symbol
        delta_symbol = self.PRODUCT_SYMBOLS.get(order.symbol, "BTCUSDT")
        if self.currency == "INR":
            delta_symbol = "BTCINR"

        # Build order payload as JSON
        import json
        payload = {
            "product_symbol": delta_symbol,
            "size": int(order.quantity * 1e8) if order.quantity < 1 else int(order.quantity),
            "side": "buy" if order.side == OrderSide.BUY else "sell",
            "order_type": self._map_order_type(order.order_type),
        }

        # Add limit price for limit orders
        if order.order_type == OrderType.LIMIT and order.price:
            payload["limit_price"] = str(order.price)

        # Add stop price for stop orders
        if order.order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT) and order.stop_price:
            payload["stop_price"] = str(order.stop_price)

        payload_str = json.dumps(payload)
        endpoint = self.ORDER_ENDPOINT
        headers = self._generate_signature("POST", endpoint, payload_str)

        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                data=payload_str,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json().get("result", {})

            # Update order with exchange response
            order.exchange_order_id = str(result.get("id", ""))
            status = result.get("state", "")
            order.status = self._map_status(status)
            order.filled_price = float(result.get("average_fill_price", 0) or order.price or 0)
            order.filled_quantity = float(result.get("size", 0)) or order.quantity

            # Delta Exchange fee: 0.05% maker / 0.15% taker (use taker as conservative)
            order.commission = order.filled_quantity * order.filled_price * 0.0015

            if order.status == OrderStatus.FILLED:
                self.filled_orders.append(order)
                self._process_fill(order)

            logger.info(
                f"Delta order placed: {order.exchange_order_id} "
                f"{order.side.value.upper()} {order.quantity} BTC/{self.currency} "
                f"Status: {order.status.value}"
            )
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_msg = e.response.json().get("error", {}).get("message", str(e))
            except Exception:
                pass
            order.reject(f"Delta rejected: {error_msg}")
            logger.error(f"Delta Exchange order rejected: {error_msg}")
        except Exception as e:
            order.reject(f"Delta error: {str(e)}")
            logger.error(f"Delta Exchange order failed: {e}")

        self.all_orders.append(order)
        return order

    def _process_fill(self, order: Order):
        """Update portfolio after a Delta Exchange order fill."""
        side = "buy" if order.side == OrderSide.BUY else "sell"
        symbol = f"BTC/{self.currency}"
        self.portfolio.open_position(
            symbol=symbol,
            side=side,
            quantity=order.filled_quantity,
            price=order.filled_price,
            commission=order.commission,
        )

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to Delta Exchange API type string."""
        mapping = {
            OrderType.MARKET: "market_order",
            OrderType.LIMIT: "limit_order",
            OrderType.STOP_LOSS: "stop_market_order",
            OrderType.TAKE_PROFIT: "stop_market_order",
        }
        return mapping.get(order_type, "market_order")

    def _map_status(self, delta_status: str) -> OrderStatus:
        """Map Delta Exchange order state to internal OrderStatus."""
        mapping = {
            "open": OrderStatus.SUBMITTED,
            "pending": OrderStatus.PENDING,
            "closed": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(delta_status, OrderStatus.PENDING)

    def cancel_order(self, order_id: str, product_id: int = 0) -> bool:
        """
        Cancel an open order on Delta Exchange.

        Args:
            order_id: Delta Exchange order ID
            product_id: Product ID (required by Delta API)

        Returns:
            True if cancellation was successful
        """
        import json
        payload = json.dumps({"id": int(order_id), "product_id": product_id})
        endpoint = self.ORDER_ENDPOINT
        headers = self._generate_signature("DELETE", endpoint, payload)

        try:
            response = requests.delete(
                f"{self.base_url}{endpoint}",
                headers=headers,
                data=payload,
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"Delta Exchange order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel Delta order {order_id}: {e}")
            return False

    def get_execution_stats(self) -> dict:
        """Return broker execution statistics."""
        return {
            "broker": "Delta Exchange",
            "mode": "LIVE",
            "currency": self.currency,
            "total_orders": len(self.all_orders),
            "filled_orders": len(self.filled_orders),
            "fill_rate": len(self.filled_orders) / max(1, len(self.all_orders)) * 100,
        }


class XMBroker:
    """
    Live broker for XM (Trading Point) via MetaTrader 5 Web API.

    XM supports both Gold (XAU/USD) and BTC/USD CFD trading.
    Accepts clients from India. Uses MT5 Manager Web API for order execution.

    XM provides MetaTrader 5 platform access — this broker connects via
    the MT5 Web API REST endpoints for automated trading.

    Supported instruments:
        - XAUUSD (Gold spot CFD)
        - BTCUSD (Bitcoin CFD)

    Requires XM_MT5_LOGIN, XM_MT5_PASSWORD, and XM_MT5_SERVER env vars.
    """

    # XM MT5 Web API base URLs (demo and live servers)
    DEMO_BASE_URL = "https://mt5-demo.xm.com/api/v1"
    LIVE_BASE_URL = "https://mt5-real.xm.com/api/v1"

    # XM instrument symbol mapping (internal symbol → MT5 symbol)
    SYMBOL_MAP = {
        "XAU/USD": "XAUUSD",     # Gold spot CFD
        "BTC/USD": "BTCUSD",     # Bitcoin CFD
        "BTC/USDT": "BTCUSD",    # Map BTC/USDT to BTC/USD on XM
    }

    def __init__(
        self,
        mt5_login: str,
        mt5_password: str,
        mt5_server: str,
        portfolio: Portfolio,
        demo: bool = True,
    ):
        """
        Initialize XM broker via MT5 Web API.

        Args:
            mt5_login: MT5 account login number
            mt5_password: MT5 account password
            mt5_server: MT5 server name (e.g., "XMGlobal-MT5")
            portfolio: Portfolio instance to track positions
            demo: If True, use demo server (recommended for testing)
        """
        self.mt5_login = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server
        self.portfolio = portfolio
        self.demo = demo

        # Select API base URL based on demo/live mode
        self.base_url = self.DEMO_BASE_URL if demo else self.LIVE_BASE_URL

        # Session token for authenticated requests (obtained via login)
        self._session_token: Optional[str] = None

        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.all_orders: List[Order] = []

        mode_label = "DEMO" if demo else "LIVE"
        logger.info(f"XM broker initialized in {mode_label} mode (server: {mt5_server})")

    def _authenticate(self) -> bool:
        """
        Authenticate with XM MT5 Web API and obtain session token.

        Returns:
            True if authentication was successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={
                    "login": self.mt5_login,
                    "password": self.mt5_password,
                    "server": self.mt5_server,
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            self._session_token = data.get("token") or data.get("session_id", "")
            logger.info("XM MT5 authentication successful")
            return True
        except Exception as e:
            logger.error(f"XM MT5 authentication failed: {e}")
            return False

    def _get_headers(self) -> dict:
        """Return authenticated request headers for XM MT5 API."""
        if not self._session_token:
            self._authenticate()
        return {
            "Authorization": f"Bearer {self._session_token}",
            "Content-Type": "application/json",
        }

    def _map_symbol(self, symbol: str) -> str:
        """Map internal trading symbol to XM MT5 symbol format."""
        return self.SYMBOL_MAP.get(symbol, symbol.replace("/", ""))

    def get_account_balance(self) -> dict:
        """
        Fetch account balance and margin info from XM MT5.

        Returns:
            Dict with account balance details (balance, equity, margin, free_margin)
        """
        try:
            response = requests.get(
                f"{self.base_url}/account/info",
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "USD": {
                    "balance": float(data.get("balance", 0)),
                    "equity": float(data.get("equity", 0)),
                    "margin": float(data.get("margin", 0)),
                    "free_margin": float(data.get("free_margin", 0)),
                    "margin_level": float(data.get("margin_level", 0)),
                    "total": float(data.get("balance", 0)),
                }
            }
        except Exception as e:
            logger.error(f"Failed to fetch XM account balance: {e}")
            return {}

    def get_price(self, symbol: str = "XAUUSD") -> float:
        """
        Fetch current market price for an XM instrument.

        Args:
            symbol: XM MT5 symbol (e.g., "XAUUSD", "BTCUSD")

        Returns:
            Current mid price as float (average of bid and ask)
        """
        mt5_symbol = self._map_symbol(symbol)
        try:
            response = requests.get(
                f"{self.base_url}/market/quote",
                params={"symbol": mt5_symbol},
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            bid = float(data.get("bid", 0))
            ask = float(data.get("ask", 0))
            # Return mid price (average of bid and ask)
            return (bid + ask) / 2 if bid and ask else bid or ask
        except Exception as e:
            logger.error(f"Failed to fetch XM price for {mt5_symbol}: {e}")
            return 0.0

    def submit_order(self, order: Order) -> Order:
        """
        Submit a trade order to XM via MT5 Web API.

        XM supports market, limit, and stop orders for Gold and BTC CFDs.
        Leveraged trading is available (check your account type for limits).

        Args:
            order: Order object to execute

        Returns:
            Updated order with XM order ID and status
        """
        mt5_symbol = self._map_symbol(order.symbol)

        # Build MT5 order payload
        payload = {
            "symbol": mt5_symbol,
            "action": "ORDER_BUY" if order.side == OrderSide.BUY else "ORDER_SELL",
            "type": self._map_order_type(order.order_type),
            "volume": round(order.quantity, 2),  # XM uses lots (0.01 = micro lot)
        }

        # Add price for limit orders
        if order.order_type == OrderType.LIMIT:
            payload["price"] = round(order.price, 5)

        # Add stop price for stop orders
        if order.order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
            payload["stop_price"] = round(order.stop_price, 5)

        # Add stop-loss and take-profit if provided via stop_price on the order
        if order.stop_price and order.order_type == OrderType.STOP_LOSS:
            payload["sl"] = round(order.stop_price, 5)
        elif order.stop_price and order.order_type == OrderType.TAKE_PROFIT:
            payload["tp"] = round(order.stop_price, 5)

        try:
            response = requests.post(
                f"{self.base_url}/trade/order",
                json=payload,
                headers=self._get_headers(),
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()

            # Update order with XM exchange data
            order.exchange_order_id = str(result.get("order_id", result.get("ticket", "")))
            order.status = self._map_status(result.get("status", ""))
            order.filled_price = float(result.get("price", 0)) or order.price
            order.filled_quantity = float(result.get("volume", 0)) or order.quantity
            # XM commission varies by account type; Standard accounts have spread-based fees
            order.commission = float(result.get("commission", 0)) or (order.filled_price * order.filled_quantity * 0.0001)

            if order.status == OrderStatus.FILLED:
                self.filled_orders.append(order)
                self._process_fill(order)

            logger.info(
                f"XM order placed: #{order.exchange_order_id} "
                f"{order.side.value.upper()} {order.quantity:.4f} {mt5_symbol} "
                f"Status: {order.status.value}"
            )
        except requests.exceptions.HTTPError as e:
            error_msg = ""
            try:
                error_msg = e.response.json().get("message", str(e))
            except Exception:
                error_msg = str(e)
            order.reject(f"XM rejected: {error_msg}")
            logger.error(f"XM order rejected: {error_msg}")
        except Exception as e:
            order.reject(f"XM error: {str(e)}")
            logger.error(f"XM order failed: {e}")

        self.all_orders.append(order)
        return order

    def _process_fill(self, order: Order):
        """Update portfolio after an XM order fill."""
        side = "buy" if order.side == OrderSide.BUY else "sell"
        self.portfolio.open_position(
            symbol=order.symbol,
            side=side,
            quantity=order.filled_quantity,
            price=order.filled_price,
            commission=order.commission,
        )

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to XM MT5 order type string."""
        mapping = {
            OrderType.MARKET: "ORDER_TYPE_BUY",     # Market execution
            OrderType.LIMIT: "ORDER_TYPE_BUY_LIMIT",
            OrderType.STOP_LOSS: "ORDER_TYPE_BUY_STOP",
            OrderType.TAKE_PROFIT: "ORDER_TYPE_BUY_LIMIT",
        }
        return mapping.get(order_type, "ORDER_TYPE_BUY")

    def _map_status(self, xm_status: str) -> OrderStatus:
        """Map XM MT5 order status to internal OrderStatus."""
        mapping = {
            "filled": OrderStatus.FILLED,
            "placed": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            # MT5 specific statuses
            "ORDER_STATE_PLACED": OrderStatus.SUBMITTED,
            "ORDER_STATE_FILLED": OrderStatus.FILLED,
            "ORDER_STATE_CANCELED": OrderStatus.CANCELLED,
            "ORDER_STATE_REJECTED": OrderStatus.REJECTED,
            "ORDER_STATE_PARTIAL": OrderStatus.PARTIALLY_FILLED,
        }
        return mapping.get(xm_status, OrderStatus.PENDING)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order on XM MT5.

        Args:
            order_id: XM MT5 order ticket number

        Returns:
            True if cancellation was successful
        """
        try:
            response = requests.delete(
                f"{self.base_url}/trade/order",
                json={"order_id": int(order_id)},
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"XM order cancelled: #{order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel XM order #{order_id}: {e}")
            return False

    def get_open_positions(self) -> List[dict]:
        """
        Fetch all open positions from XM MT5.

        Returns:
            List of open position dicts with symbol, volume, profit, etc.
        """
        try:
            response = requests.get(
                f"{self.base_url}/trade/positions",
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            positions = response.json()
            return [
                {
                    "ticket": p.get("ticket"),
                    "symbol": p.get("symbol"),
                    "volume": float(p.get("volume", 0)),
                    "type": p.get("type"),  # BUY or SELL
                    "price_open": float(p.get("price_open", 0)),
                    "price_current": float(p.get("price_current", 0)),
                    "profit": float(p.get("profit", 0)),
                    "sl": float(p.get("sl", 0)),
                    "tp": float(p.get("tp", 0)),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to fetch XM open positions: {e}")
            return []

    def close_position(self, ticket: int) -> bool:
        """
        Close an open position on XM MT5.

        Args:
            ticket: Position ticket number to close

        Returns:
            True if position was closed successfully
        """
        try:
            response = requests.post(
                f"{self.base_url}/trade/close",
                json={"ticket": ticket},
                headers=self._get_headers(),
                timeout=15,
            )
            response.raise_for_status()
            logger.info(f"XM position closed: #{ticket}")
            return True
        except Exception as e:
            logger.error(f"Failed to close XM position #{ticket}: {e}")
            return False

    def get_execution_stats(self) -> dict:
        """Return broker execution statistics."""
        return {
            "broker": "XM (MT5)",
            "mode": "DEMO" if self.demo else "LIVE",
            "server": self.mt5_server,
            "total_orders": len(self.all_orders),
            "filled_orders": len(self.filled_orders),
            "fill_rate": len(self.filled_orders) / max(1, len(self.all_orders)) * 100,
        }
