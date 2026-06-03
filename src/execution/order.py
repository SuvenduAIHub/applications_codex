"""
Order model and order book management.
Defines order data structures and lifecycle for the execution engine.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config.settings import OrderSide, OrderStatus, OrderType
from src.utils.helpers import generate_order_id


@dataclass
class Order:
    """
    Represents a trading order with full lifecycle tracking.
    Covers market, limit, stop-loss, and take-profit order types.
    """
    symbol: str                                          # Trading pair (e.g., "BTC/USDT")
    side: OrderSide                                      # Buy or Sell
    order_type: OrderType                                # Market, Limit, StopLoss, etc.
    quantity: float                                      # Quantity in base asset units
    price: Optional[float] = None                        # Limit price (None for market)
    stop_price: Optional[float] = None                   # Trigger price for stop orders
    status: OrderStatus = OrderStatus.PENDING            # Current order status
    order_id: str = ""                                   # Unique identifier
    filled_quantity: float = 0.0                         # Amount filled so far
    filled_price: float = 0.0                            # Average fill price
    commission: float = 0.0                              # Total commission paid
    slippage: float = 0.0                                # Actual slippage experienced
    strategy: str = ""                                   # Strategy that created this order
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None                 # When the order was fully filled
    notes: str = ""                                      # Additional metadata

    def __post_init__(self):
        """Generate a unique order ID if not provided."""
        if not self.order_id:
            self.order_id = generate_order_id(self.symbol, self.side.value)

    @property
    def is_active(self) -> bool:
        """Check if the order is still active (pending or partially filled)."""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_filled(self) -> bool:
        """Check if the order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> float:
        """Calculate the unfilled portion of the order."""
        return max(0, self.quantity - self.filled_quantity)

    @property
    def notional_value(self) -> float:
        """Calculate the USD value of the order based on fill price."""
        if self.filled_price > 0:
            return self.filled_quantity * self.filled_price
        elif self.price:
            return self.quantity * self.price
        return 0.0

    def fill(self, fill_price: float, fill_quantity: Optional[float] = None, commission: float = 0.0) -> None:
        """
        Record a fill event for this order.

        Args:
            fill_price: Price at which the fill occurred
            fill_quantity: Amount filled (defaults to remaining quantity)
            commission: Commission charged for this fill
        """
        qty = fill_quantity or self.remaining_quantity

        # Calculate weighted average fill price
        total_filled_value = self.filled_price * self.filled_quantity + fill_price * qty
        self.filled_quantity += qty
        self.filled_price = total_filled_value / self.filled_quantity if self.filled_quantity > 0 else 0
        self.commission += commission

        self.updated_at = datetime.now(timezone.utc)

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = self.updated_at
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Cancel this order."""
        if self.is_active:
            self.status = OrderStatus.CANCELLED
            self.updated_at = datetime.now(timezone.utc)

    def reject(self, reason: str = "") -> None:
        """Reject this order with an optional reason."""
        self.status = OrderStatus.REJECTED
        self.notes = reason
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert order to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "strategy": self.strategy,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "notes": self.notes,
        }
