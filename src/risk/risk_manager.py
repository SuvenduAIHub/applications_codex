"""
Core risk management module.
Enforces portfolio-level risk constraints including drawdown limits,
position limits, daily loss caps, and asset allocation rules.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from config.settings import RiskConfig
from src.risk.position_sizer import PositionSizer


class RiskManager:
    """
    Central risk management engine that validates trades and monitors
    portfolio health in real-time.

    Enforced constraints:
        - Maximum risk per trade
        - Maximum portfolio exposure
        - Maximum concurrent positions
        - Maximum drawdown (circuit breaker)
        - Daily loss limit
        - Consecutive loss cooldown
        - Per-asset allocation limits
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize the risk manager.

        Args:
            config: Risk configuration parameters
        """
        self.config = config or RiskConfig()
        self.position_sizer = PositionSizer(self.config)

        # Portfolio state tracking
        self.peak_portfolio_value: float = 0.0
        self.current_portfolio_value: float = 0.0
        self.daily_pnl: float = 0.0
        self.daily_start_value: float = 0.0
        self.consecutive_losses: int = 0
        self.last_loss_streak_cooldown_start: Optional[datetime] = None

        # Open position tracking
        self.open_positions: Dict[str, dict] = {}

        # Trade history for the current day
        self.daily_trades: List[dict] = []

        # Circuit breaker flag - halts all trading when True
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.symbol_halts: Dict[str, str] = {}

    def initialize(self, portfolio_value: float) -> None:
        """
        Initialize the risk manager with the current portfolio value.
        Should be called at system startup and at the start of each day.

        Args:
            portfolio_value: Current total portfolio value (USD)
        """
        self.current_portfolio_value = portfolio_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        self.daily_start_value = portfolio_value
        self.daily_pnl = 0.0
        self.daily_trades = []
        logger.info(f"Risk manager initialized: portfolio=${portfolio_value:,.2f}")

    def update_portfolio_value(self, portfolio_value: float) -> None:
        """
        Refresh mark-to-market portfolio value used by exposure and drawdown checks.

        Args:
            portfolio_value: Current total portfolio value (USD)
        """
        self.current_portfolio_value = portfolio_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)

    def can_trade(self, symbol: str, side: str, size_usd: float) -> tuple:
        """
        Validate whether a proposed trade passes all risk checks.

        Args:
            symbol: Trading pair symbol
            side: "buy" or "sell"
            size_usd: Proposed position size in USD

        Returns:
            Tuple of (is_allowed: bool, reason: str)
        """
        # Check circuit breaker
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"

        if symbol in self.symbol_halts:
            return False, f"{symbol} halted: {self.symbol_halts[symbol]}"

        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl / self.daily_start_value * 100) if self.daily_start_value > 0 else 0
        if self.daily_pnl < 0 and daily_loss_pct >= self.config.max_daily_loss_pct:
            return False, f"Daily loss limit reached: {daily_loss_pct:.2f}% >= {self.config.max_daily_loss_pct}%"

        # Check maximum concurrent positions
        if len(self.open_positions) >= self.config.max_concurrent_positions:
            # Allow if closing an existing position
            if symbol not in self.open_positions or side == self.open_positions[symbol].get("side"):
                return False, f"Max concurrent positions ({self.config.max_concurrent_positions}) reached"

        # Check maximum drawdown (circuit breaker)
        current_drawdown = self._calculate_drawdown()
        if abs(current_drawdown) >= self.config.max_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown breached: {current_drawdown:.2f}%"
            return False, self.halt_reason

        # Check consecutive loss cooldown
        if self._in_loss_cooldown():
            return False, f"In loss streak cooldown ({self.consecutive_losses} consecutive losses)"

        # Check portfolio exposure
        total_exposure = sum(
            pos.get("size_usd", 0) for pos in self.open_positions.values()
        )
        if total_exposure + size_usd > self.current_portfolio_value * (self.config.max_portfolio_exposure_pct / 100):
            return False, (
                f"Portfolio exposure limit: {total_exposure + size_usd:.2f} > "
                f"{self.current_portfolio_value * self.config.max_portfolio_exposure_pct / 100:.2f}"
            )

        # Check per-asset allocation limit
        if symbol in self.config.asset_allocation_limits:
            limit_pct = self.config.asset_allocation_limits[symbol]
            max_allocation = self.current_portfolio_value * (limit_pct / 100)
            current_allocation = self.open_positions.get(symbol, {}).get("size_usd", 0)
            if current_allocation + size_usd > max_allocation:
                return False, (
                    f"Asset allocation limit for {symbol}: "
                    f"{current_allocation + size_usd:.2f} > {max_allocation:.2f}"
                )

        # Check minimum order size
        if size_usd < 10.0:
            return False, f"Order size ${size_usd:.2f} below minimum $10.00"

        return True, "Trade approved"

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: float = 0.0,
        custom_pct: Optional[float] = None,
        position_notional_usd: Optional[float] = None,
    ) -> float:
        """
        Calculate stop-loss price for a position.

        Args:
            entry_price: Trade entry price
            side: "buy" (long) or "sell" (short)
            atr: Current ATR for dynamic stop calculation
            custom_pct: Override stop-loss percentage
            position_notional_usd: Leveraged position value for dollar-PnL stops

        Returns:
            Stop-loss price
        """
        pct = custom_pct or self.config.default_stop_loss_pct

        if self.config.fixed_stop_loss_usd > 0 and position_notional_usd and position_notional_usd > 0:
            quantity = position_notional_usd / entry_price
            stop_distance = self.config.fixed_stop_loss_usd / quantity
        elif self.config.fixed_stop_loss_usd > 0:
            stop_distance = self.config.fixed_stop_loss_usd
        elif atr > 0:
            # ATR-based stop: 2x ATR from entry
            stop_distance = 2.0 * atr
        else:
            # Percentage-based stop
            stop_distance = entry_price * (pct / 100.0)

        if side == "buy":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        stop_loss: float,
        custom_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate take-profit price ensuring minimum risk-reward ratio.

        Args:
            entry_price: Trade entry price
            side: "buy" or "sell"
            stop_loss: Stop-loss price
            custom_pct: Override take-profit percentage

        Returns:
            Take-profit price
        """
        pct = custom_pct or self.config.default_take_profit_pct

        # Ensure minimum risk-reward ratio
        risk = abs(entry_price - stop_loss)
        min_reward = risk * self.config.min_risk_reward_ratio

        if side == "buy":
            tp_by_pct = entry_price * (1 + pct / 100.0)
            tp_by_rr = entry_price + min_reward
            return max(tp_by_pct, tp_by_rr)
        else:
            tp_by_pct = entry_price * (1 - pct / 100.0)
            tp_by_rr = entry_price - min_reward
            return min(tp_by_pct, tp_by_rr)

    def register_position(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float],
        notional_usd: Optional[float] = None,
    ) -> None:
        """
        Register a new open position for tracking.

        Args:
            symbol: Trading pair symbol
            side: Position direction ("buy"/"sell")
            size_usd: Position size in USD
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            notional_usd: Leveraged position value used for actual PnL dollar stops
        """
        position_notional = notional_usd if notional_usd is not None else size_usd
        self.open_positions[symbol] = {
            "side": side,
            "size_usd": size_usd,
            "notional_usd": position_notional,
            "quantity": position_notional / entry_price if entry_price > 0 else 0.0,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opened_at": datetime.now(timezone.utc),
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "best_pnl_usd": 0.0,
        }
        tp_text = f"{take_profit:.2f}" if take_profit is not None else "disabled"
        logger.info(
            f"Position registered: {side.upper()} {symbol} "
            f"${size_usd:,.2f} @ {entry_price:.2f} "
            f"(SL={stop_loss:.2f}, TP={tp_text})"
        )

    def close_position(self, symbol: str, exit_price: float) -> Optional[dict]:
        """
        Close an open position and record the PnL.

        Args:
            symbol: Trading pair symbol
            exit_price: Exit/closing price

        Returns:
            Dict with position result details, or None if no position exists
        """
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions.pop(symbol)
        entry = pos["entry_price"]
        size = pos["size_usd"]
        notional = pos.get("notional_usd", size)
        side = pos["side"]

        # Calculate PnL
        if side == "buy":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        pnl_usd = notional * pnl_pct

        # Update tracking
        self.daily_pnl += pnl_usd
        self.current_portfolio_value += pnl_usd
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.current_portfolio_value)

        # Track consecutive losses
        if pnl_usd < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                self.last_loss_streak_cooldown_start = datetime.now(timezone.utc)
                logger.warning(
                    f"Consecutive loss limit hit ({self.consecutive_losses}). "
                    f"Entering {self.config.loss_streak_cooldown_minutes}min cooldown."
                )
        else:
            self.consecutive_losses = 0

        result = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "size_usd": size,
            "notional_usd": notional,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct * 100,
            "duration": (datetime.now(timezone.utc) - pos["opened_at"]).total_seconds(),
        }

        logger.info(
            f"Position closed: {symbol} PnL=${pnl_usd:,.2f} ({pnl_pct * 100:.2f}%)"
        )
        return result

    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Update trailing stop-loss for an open position.
        Moves the stop-loss up (for longs) when price advances.

        Args:
            symbol: Trading pair symbol
            current_price: Current market price

        Returns:
            New stop-loss price if updated, None otherwise
        """
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]
        entry = pos["entry_price"]
        side = pos["side"]
        activation_pct = self.config.trailing_stop_activation_pct / 100
        distance_pct = self.config.trailing_stop_distance_pct / 100
        activation_usd = self.config.trailing_stop_activation_usd
        distance_usd = self.config.trailing_stop_distance_usd
        quantity = pos.get("quantity", 0.0) or (
            pos.get("notional_usd", 0.0) / entry if entry > 0 else 0.0
        )

        if side == "buy":
            # Update highest price seen
            pos["highest_price"] = max(pos["highest_price"], current_price)
            if activation_usd > 0 and distance_usd > 0:
                current_pnl = (current_price - entry) * quantity
                should_trail = current_pnl >= activation_usd
                new_stop = pos["highest_price"] - (distance_usd / quantity if quantity > 0 else distance_usd)
            else:
                profit_pct = (current_price - entry) / entry
                should_trail = profit_pct >= activation_pct
                new_stop = pos["highest_price"] * (1 - distance_pct)
            if should_trail:
                if new_stop > pos["stop_loss"]:
                    pos["stop_loss"] = new_stop
                    logger.debug(f"Trailing stop updated for {symbol}: {new_stop:.2f}")
                    return new_stop
        else:
            pos["lowest_price"] = min(pos["lowest_price"], current_price)
            if activation_usd > 0 and distance_usd > 0:
                current_pnl = (entry - current_price) * quantity
                should_trail = current_pnl >= activation_usd
                new_stop = pos["lowest_price"] + (distance_usd / quantity if quantity > 0 else distance_usd)
            else:
                profit_pct = (entry - current_price) / entry
                should_trail = profit_pct >= activation_pct
                new_stop = pos["lowest_price"] * (1 + distance_pct)
            if should_trail:
                if new_stop < pos["stop_loss"]:
                    pos["stop_loss"] = new_stop
                    logger.debug(f"Trailing stop updated for {symbol}: {new_stop:.2f}")
                    return new_stop

        return None

    def check_stop_levels(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if current price has hit stop-loss or take-profit.

        Args:
            symbol: Trading pair
            current_price: Current market price

        Returns:
            "stop_loss", "take_profit", or None
        """
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]
        entry = pos["entry_price"]
        quantity = pos.get("quantity", 0.0) or (
            pos.get("notional_usd", 0.0) / entry if entry > 0 else 0.0
        )

        if pos["side"] == "buy":
            pos["highest_price"] = max(pos["highest_price"], current_price)
            current_pnl = (current_price - entry) * quantity
        else:
            pos["lowest_price"] = min(pos["lowest_price"], current_price)
            current_pnl = (entry - current_price) * quantity

        pos["best_pnl_usd"] = max(pos.get("best_pnl_usd", 0.0), current_pnl)
        giveback_activation = self.config.profit_giveback_activation_usd
        giveback_exit_loss = self.config.profit_giveback_exit_loss_usd
        if (
            giveback_activation > 0
            and giveback_exit_loss > 0
            and pos["best_pnl_usd"] >= giveback_activation
            and current_pnl <= -giveback_exit_loss
        ):
            logger.info(
                f"Profit giveback exit for {symbol}: best=${pos['best_pnl_usd']:,.2f}, "
                f"current=${current_pnl:,.2f}"
            )
            return "profit_giveback"

        if pos["side"] == "buy":
            if current_price <= pos["stop_loss"]:
                return "stop_loss"
            if pos.get("take_profit") is not None and current_price >= pos["take_profit"]:
                return "take_profit"
        else:
            if current_price >= pos["stop_loss"]:
                return "stop_loss"
            if pos.get("take_profit") is not None and current_price <= pos["take_profit"]:
                return "take_profit"

        return None

    def get_portfolio_risk_parity_allocation(
        self,
        btc_volatility: float,
        gold_volatility: float,
    ) -> Dict[str, float]:
        """
        Calculate risk-parity allocation between BTC and Gold.
        Each asset is weighted inversely to its volatility so that
        both contribute equally to total portfolio risk.

        Args:
            btc_volatility: BTC annualized volatility
            gold_volatility: Gold annualized volatility

        Returns:
            Dict with allocation percentages: {"BTC/USDT": pct, "XAU/USD": pct}
        """
        if btc_volatility <= 0 or gold_volatility <= 0:
            return {"BTC/USDT": 50.0, "XAU/USD": 50.0}

        # Inverse volatility weighting
        inv_btc = 1.0 / btc_volatility
        inv_gold = 1.0 / gold_volatility
        total_inv = inv_btc + inv_gold

        btc_alloc = (inv_btc / total_inv) * 100
        gold_alloc = (inv_gold / total_inv) * 100

        # Enforce per-asset allocation limits from config
        btc_limit = self.config.asset_allocation_limits.get("BTC/USDT", 100.0)
        gold_limit = self.config.asset_allocation_limits.get("XAU/USD", 100.0)

        btc_alloc = min(btc_alloc, btc_limit)
        gold_alloc = min(gold_alloc, gold_limit)

        logger.info(
            f"Risk parity allocation: BTC={btc_alloc:.1f}%, Gold={gold_alloc:.1f}% "
            f"(BTC vol={btc_volatility:.2f}, Gold vol={gold_volatility:.2f})"
        )
        return {"BTC/USDT": btc_alloc, "XAU/USD": gold_alloc}

    def get_risk_summary(self) -> dict:
        """
        Return a summary of current risk metrics.

        Returns:
            Dict with key risk metrics for monitoring
        """
        total_exposure = sum(
            p.get("size_usd", 0) for p in self.open_positions.values()
        )
        total_notional_exposure = sum(
            p.get("notional_usd", p.get("size_usd", 0)) for p in self.open_positions.values()
        )
        max_exposure = self.current_portfolio_value * (self.config.max_portfolio_exposure_pct / 100)
        return {
            "portfolio_value": self.current_portfolio_value,
            "peak_value": self.peak_portfolio_value,
            "drawdown_pct": self._calculate_drawdown(),
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": (self.daily_pnl / self.daily_start_value * 100
                              if self.daily_start_value > 0 else 0),
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
            "max_daily_loss_usd": self.daily_start_value * (self.config.max_daily_loss_pct / 100),
            "open_positions": len(self.open_positions),
            "total_exposure_usd": total_exposure,
            "total_notional_exposure_usd": total_notional_exposure,
            "max_exposure_pct": self.config.max_portfolio_exposure_pct,
            "max_exposure_usd": max_exposure,
            "exposure_used_pct": (total_exposure / max_exposure * 100 if max_exposure > 0 else 0),
            "consecutive_losses": self.consecutive_losses,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "symbol_halts": self.symbol_halts,
        }

    def halt_symbol(self, symbol: str, reason: str = "Manual symbol halt") -> None:
        """Prevent new trades for one symbol while keeping the rest of the system active."""
        self.symbol_halts[symbol] = reason
        logger.warning(f"Trading halted for {symbol}: {reason}")

    def resume_symbol(self, symbol: str) -> None:
        """Allow trading again for one manually halted symbol."""
        if symbol in self.symbol_halts:
            del self.symbol_halts[symbol]
            logger.info(f"Trading resumed for {symbol}")

    def is_symbol_halted(self, symbol: str) -> bool:
        """Return whether new trades are blocked for a symbol."""
        return symbol in self.symbol_halts

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak as a percentage."""
        if self.peak_portfolio_value <= 0:
            return 0.0
        dd = (self.current_portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value * 100
        return dd

    def _in_loss_cooldown(self) -> bool:
        """Check if the system is in a consecutive-loss cooldown period."""
        if self.last_loss_streak_cooldown_start is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.last_loss_streak_cooldown_start).total_seconds()
        cooldown_seconds = self.config.loss_streak_cooldown_minutes * 60
        if elapsed >= cooldown_seconds:
            self.last_loss_streak_cooldown_start = None
            self.consecutive_losses = 0
            return False
        return True

    def reset(self) -> None:
        """Reset all risk manager state (for backtesting)."""
        self.peak_portfolio_value = 0.0
        self.current_portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.daily_start_value = 0.0
        self.consecutive_losses = 0
        self.last_loss_streak_cooldown_start = None
        self.open_positions = {}
        self.daily_trades = []
        self.trading_halted = False
        self.halt_reason = None
        self.symbol_halts = {}
