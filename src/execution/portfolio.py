"""
Portfolio management module.
Tracks positions, balances, and overall portfolio state
across both BTC/USDT and XAU/USD trading pairs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from config.settings import PositionSide


@dataclass
class Position:
    """Represents an open position in a trading pair."""
    symbol: str                          # Trading pair
    side: PositionSide                   # Long, Short, or Flat
    quantity: float = 0.0                # Position size in base units
    entry_price: float = 0.0            # Average entry price
    current_price: float = 0.0          # Latest market price
    unrealized_pnl: float = 0.0         # Unrealized profit/loss (USD)
    realized_pnl: float = 0.0           # Realized profit/loss (USD)
    stop_loss: Optional[float] = None   # Current stop-loss price
    take_profit: Optional[float] = None # Current take-profit price
    opened_at: Optional[datetime] = None

    @property
    def market_value(self) -> float:
        """Signed current market value of the position."""
        value = self.quantity * self.current_price
        if self.side == PositionSide.SHORT:
            return -value
        return value

    @property
    def cost_basis(self) -> float:
        """Total cost of the position at entry."""
        return self.quantity * self.entry_price

    @property
    def pnl_pct(self) -> float:
        """Unrealized PnL as a percentage of cost basis."""
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        elif self.side == PositionSide.SHORT:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
        return 0.0

    def update_price(self, price: float) -> None:
        """Update the position with the latest market price."""
        self.current_price = price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity


class Portfolio:
    """
    Manages the overall trading portfolio including cash balance,
    open positions, and trade history.
    """

    def __init__(self, initial_balance: float = 100000.0):
        """
        Initialize the portfolio.

        Args:
            initial_balance: Starting cash balance in USD
        """
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[dict] = []
        self.equity_curve: List[dict] = []

        logger.info(f"Portfolio initialized with ${initial_balance:,.2f}")

    @property
    def total_value(self) -> float:
        """Total portfolio value: cash + sum of all position market values."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value

    @property
    def total_unrealized_pnl(self) -> float:
        """Sum of unrealized PnL across all open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def total_realized_pnl(self) -> float:
        """Sum of all realized PnL from closed trades."""
        return sum(t.get("pnl", 0) for t in self.trade_history)

    @property
    def total_return_pct(self) -> float:
        """Total return percentage from initial balance."""
        if self.initial_balance == 0:
            return 0.0
        return ((self.total_value - self.initial_balance) / self.initial_balance) * 100

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        commission: float = 0.0,
    ) -> Position:
        """
        Open a new position or add to an existing one.

        Args:
            symbol: Trading pair symbol
            side: "buy" (long) or "sell" (short)
            quantity: Amount in base asset units
            price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            commission: Transaction commission

        Returns:
            The opened/updated Position object
        """
        pos_side = PositionSide.LONG if side == "buy" else PositionSide.SHORT
        notional = quantity * price

        if pos_side == PositionSide.LONG:
            cash_delta = -(notional + commission)
        else:
            cash_delta = notional - commission
        self.cash_balance += cash_delta

        if symbol in self.positions and self.positions[symbol].side == pos_side:
            # Add to existing position (average price)
            existing = self.positions[symbol]
            total_qty = existing.quantity + quantity
            avg_price = (
                (existing.entry_price * existing.quantity + price * quantity) / total_qty
            )
            existing.quantity = total_qty
            existing.entry_price = avg_price
            existing.stop_loss = stop_loss or existing.stop_loss
            existing.take_profit = take_profit or existing.take_profit
            existing.update_price(price)
            pos = existing
        else:
            # Create new position
            pos = Position(
                symbol=symbol,
                side=pos_side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=datetime.now(timezone.utc),
            )
            self.positions[symbol] = pos

        logger.info(
            f"Position opened: {side.upper()} {quantity:.6f} {symbol} "
            f"@ {price:.2f} (notional=${notional:,.2f})"
        )
        return pos

    def close_position(
        self,
        symbol: str,
        price: float,
        quantity: Optional[float] = None,
        commission: float = 0.0,
        reason: str = "",
    ) -> Optional[dict]:
        """
        Close (or partially close) a position.

        Args:
            symbol: Trading pair symbol
            price: Exit price
            quantity: Amount to close (None = close all)
            commission: Transaction commission
            reason: Reason for closing (e.g., "stop_loss", "take_profit", "signal")

        Returns:
            Trade result dict with PnL details, or None if no position exists
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        close_qty = quantity or pos.quantity

        # Calculate PnL
        if pos.side == PositionSide.LONG:
            pnl = (price - pos.entry_price) * close_qty - commission
        else:
            pnl = (pos.entry_price - price) * close_qty - commission

        # Settle cash. Long closes receive proceeds; short closes buy back borrowed units.
        notional = close_qty * price
        if pos.side == PositionSide.LONG:
            self.cash_balance += notional - commission
        else:
            self.cash_balance -= notional + commission

        # Build trade record
        trade_record = {
            "symbol": symbol,
            "side": "buy" if pos.side == PositionSide.LONG else "sell",
            "quantity": close_qty,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "pnl": pnl,
            "pnl_pct": (pnl / (close_qty * pos.entry_price)) * 100 if pos.entry_price > 0 else 0,
            "commission": commission,
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": (
                (datetime.now(timezone.utc) - pos.opened_at).total_seconds()
                if pos.opened_at else 0
            ),
        }
        self.trade_history.append(trade_record)

        # Update or remove position
        if close_qty >= pos.quantity:
            del self.positions[symbol]
        else:
            pos.quantity -= close_qty
            pos.realized_pnl += pnl

        logger.info(
            f"Position closed: {symbol} PnL=${pnl:,.2f} "
            f"({trade_record['pnl_pct']:.2f}%) - {reason}"
        )
        return trade_record

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update all position prices with latest market data.

        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def record_equity(self) -> None:
        """Record current equity curve data point."""
        self.equity_curve.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_value": self.total_value,
            "cash": self.cash_balance,
            "positions_value": sum(p.market_value for p in self.positions.values()),
            "unrealized_pnl": self.total_unrealized_pnl,
        })

    def get_summary(self, inr_rate: float = 0.0) -> dict:
        """
        Return a summary of portfolio state with optional dual-currency display.

        Args:
            inr_rate: Current INR per USD rate. If > 0, includes INR values.

        Returns:
            Portfolio summary dict with USD values (and INR if rate provided)
        """
        summary = {
            "total_value": self.total_value,
            "cash_balance": self.cash_balance,
            "initial_balance": self.initial_balance,
            "total_return_pct": self.total_return_pct,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "open_positions": {
                s: {
                    "side": p.side.value,
                    "qty": p.quantity,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.pnl_pct,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "entry_time": p.opened_at.isoformat() if p.opened_at else None,
                    "source": getattr(p, "source", "Auto"),
                }
                for s, p in self.positions.items()
            },
            "total_trades": len(self.trade_history),
            "currency": "USD",
        }

        # Add INR-converted values if exchange rate is provided
        if inr_rate > 0:
            summary["inr"] = {
                "total_value": self.total_value * inr_rate,
                "cash_balance": self.cash_balance * inr_rate,
                "initial_balance": self.initial_balance * inr_rate,
                "total_realized_pnl": self.total_realized_pnl * inr_rate,
                "total_unrealized_pnl": self.total_unrealized_pnl * inr_rate,
                "exchange_rate": inr_rate,
            }
            summary["usd"] = {
                "total_value": self.total_value,
                "cash_balance": self.cash_balance,
                "total_realized_pnl": self.total_realized_pnl,
                "total_unrealized_pnl": self.total_unrealized_pnl,
            }

        return summary

    def reset(self) -> None:
        """Reset portfolio to initial state (for backtesting)."""
        self.cash_balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
