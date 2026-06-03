"""
Simulated broker for paper trading and backtesting.
Handles order submission, fill simulation with slippage and commission,
and order lifecycle management. Can be extended to connect to real exchanges.
"""

import random
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from config.settings import ExecutionConfig, OrderSide, OrderStatus, OrderType
from src.execution.order import Order
from src.execution.portfolio import Portfolio


class SimulatedBroker:
    """
    Simulated broker that models realistic order execution.

    Features:
        - Market, limit, and stop order support
        - Configurable slippage simulation
        - Commission/fee calculation
        - Order book management with pending/filled/cancelled tracking
        - Latency simulation for paper trading mode

    For live trading, this class would be extended with real exchange API calls.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        config: Optional[ExecutionConfig] = None,
    ):
        """
        Initialize the simulated broker.

        Args:
            portfolio: Portfolio instance to update on fills
            config: Execution configuration for slippage/commission
        """
        self.portfolio = portfolio
        self.config = config or ExecutionConfig()

        # Order tracking
        self.active_orders: Dict[str, Order] = {}     # Orders awaiting fill
        self.filled_orders: List[Order] = []           # Completed orders
        self.cancelled_orders: List[Order] = []        # Cancelled orders
        self.all_orders: List[Order] = []              # Full order history

    def submit_order(self, order: Order) -> Order:
        """
        Submit a new order for execution.

        Args:
            order: Order to submit

        Returns:
            The order with updated status
        """
        # Validate order size limits
        notional = (order.price or 0) * order.quantity
        if order.order_type == OrderType.MARKET:
            # Market orders are filled immediately in the next process_tick
            order.status = OrderStatus.SUBMITTED
            self.active_orders[order.order_id] = order
        elif order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if order.price is None:
                order.reject("Limit order requires a price")
                logger.warning(f"Order {order.order_id} rejected: no price specified")
                return order
            order.status = OrderStatus.SUBMITTED
            self.active_orders[order.order_id] = order
        elif order.order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
            if order.stop_price is None:
                order.reject("Stop order requires a stop price")
                return order
            order.status = OrderStatus.SUBMITTED
            self.active_orders[order.order_id] = order
        else:
            order.reject(f"Unsupported order type: {order.order_type}")
            return order

        self.all_orders.append(order)
        logger.info(
            f"Order submitted: {order.order_id} "
            f"{order.side.value.upper()} {order.quantity:.6f} {order.symbol} "
            f"@ {order.price or 'MARKET'}"
        )
        return order

    def process_tick(
        self,
        symbol: str,
        current_price: float,
        high: float = 0,
        low: float = 0,
    ) -> List[Order]:
        """
        Process all pending orders against the current market tick.
        Checks if any orders should be filled based on current price.

        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            high: High price of current candle (for limit order matching)
            low: Low price of current candle (for limit order matching)

        Returns:
            List of orders that were filled during this tick
        """
        if high == 0:
            high = current_price
        if low == 0:
            low = current_price

        filled_this_tick = []
        orders_to_remove = []

        for order_id, order in self.active_orders.items():
            if order.symbol != symbol:
                continue

            should_fill = False
            fill_price = current_price

            if order.order_type == OrderType.MARKET:
                # Market orders fill immediately at current price + slippage
                should_fill = True
                fill_price = self._apply_slippage(current_price, order.side)

            elif order.order_type == OrderType.LIMIT:
                # Limit buy fills when price drops to/below limit
                if order.side == OrderSide.BUY and low <= order.price:
                    should_fill = True
                    fill_price = order.price
                # Limit sell fills when price rises to/above limit
                elif order.side == OrderSide.SELL and high >= order.price:
                    should_fill = True
                    fill_price = order.price

            elif order.order_type == OrderType.STOP_LOSS:
                # Stop-loss buy triggers when price rises above stop price
                if order.side == OrderSide.BUY and high >= order.stop_price:
                    should_fill = True
                    fill_price = self._apply_slippage(order.stop_price, order.side)
                # Stop-loss sell triggers when price drops below stop price
                elif order.side == OrderSide.SELL and low <= order.stop_price:
                    should_fill = True
                    fill_price = self._apply_slippage(order.stop_price, order.side)

            elif order.order_type == OrderType.TAKE_PROFIT:
                # Take-profit buy triggers when price drops to target
                if order.side == OrderSide.BUY and low <= order.stop_price:
                    should_fill = True
                    fill_price = order.stop_price
                # Take-profit sell triggers when price rises to target
                elif order.side == OrderSide.SELL and high >= order.stop_price:
                    should_fill = True
                    fill_price = order.stop_price

            if should_fill:
                commission = self._calculate_commission(fill_price, order.quantity)
                order.fill(fill_price, commission=commission)
                order.slippage = abs(fill_price - (order.price or current_price))
                filled_this_tick.append(order)
                orders_to_remove.append(order_id)
                self.filled_orders.append(order)

                logger.info(
                    f"Order filled: {order.order_id} {order.side.value.upper()} "
                    f"{order.quantity:.6f} {order.symbol} @ {fill_price:.2f} "
                    f"(commission=${commission:.2f})"
                )

        # Remove filled orders from active list
        for oid in orders_to_remove:
            del self.active_orders[oid]

        return filled_this_tick

    def cancel_order(self, order_id: str) -> Optional[Order]:
        """
        Cancel a pending order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            The cancelled order, or None if not found
        """
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            order.cancel()
            self.cancelled_orders.append(order)
            logger.info(f"Order cancelled: {order_id}")
            return order
        return None

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active orders, optionally filtered by symbol.

        Args:
            symbol: Optional filter to cancel only orders for this symbol

        Returns:
            Number of orders cancelled
        """
        to_cancel = []
        for oid, order in self.active_orders.items():
            if symbol is None or order.symbol == symbol:
                to_cancel.append(oid)

        for oid in to_cancel:
            self.cancel_order(oid)

        return len(to_cancel)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active (pending) orders, optionally filtered by symbol."""
        orders = list(self.active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        Apply simulated slippage to a fill price.
        Slippage is always adverse: higher for buys, lower for sells.

        Args:
            price: Base price before slippage
            side: Order side (buy/sell)

        Returns:
            Price with slippage applied
        """
        # Random slippage between 0 and configured max
        slippage_pct = random.uniform(0, self.config.simulated_slippage_pct) / 100.0

        if side == OrderSide.BUY:
            return price * (1 + slippage_pct)  # Buys fill higher
        else:
            return price * (1 - slippage_pct)  # Sells fill lower

    def _calculate_commission(self, price: float, quantity: float) -> float:
        """
        Calculate the commission for a trade.

        Args:
            price: Fill price
            quantity: Fill quantity

        Returns:
            Commission amount in USD
        """
        notional = price * quantity
        return notional * (self.config.simulated_commission_pct / 100.0)

    def get_execution_stats(self) -> dict:
        """Return statistics about order execution."""
        total_orders = len(self.all_orders)
        total_filled = len(self.filled_orders)
        total_cancelled = len(self.cancelled_orders)
        total_commission = sum(o.commission for o in self.filled_orders)
        total_slippage = sum(o.slippage for o in self.filled_orders)

        return {
            "total_orders": total_orders,
            "filled_orders": total_filled,
            "cancelled_orders": total_cancelled,
            "active_orders": len(self.active_orders),
            "fill_rate": total_filled / total_orders if total_orders > 0 else 0,
            "total_commission_usd": total_commission,
            "total_slippage_usd": total_slippage,
            "avg_slippage_pct": (
                sum(o.slippage / o.filled_price * 100 for o in self.filled_orders if o.filled_price > 0)
                / total_filled if total_filled > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Reset broker state (for backtesting)."""
        self.active_orders = {}
        self.filled_orders = []
        self.cancelled_orders = []
        self.all_orders = []
