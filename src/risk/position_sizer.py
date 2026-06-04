"""
Position sizing module implementing multiple sizing strategies.
Determines how much capital to allocate per trade based on
risk tolerance, volatility, and portfolio state.
"""

import math
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import RiskConfig


class PositionSizer:
    """
    Calculates position sizes using multiple methods:
    - Fixed percentage of portfolio
    - Kelly Criterion (optimal fraction)
    - Volatility-based (ATR-adjusted)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize the position sizer.

        Args:
            config: Risk configuration with sizing parameters
        """
        self.config = config or RiskConfig()

    def fixed_percentage(
        self,
        portfolio_value: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate position size using fixed percentage of portfolio.

        Args:
            portfolio_value: Current total portfolio value (USD)
            risk_pct: Percentage to risk (default from config)

        Returns:
            Maximum position size in USD
        """
        pct = risk_pct or self.config.max_risk_per_trade_pct
        position_size = portfolio_value * (pct / 100.0)
        logger.debug(f"Fixed % sizing: {pct}% of ${portfolio_value:,.2f} = ${position_size:,.2f}")
        return position_size

    def kelly_criterion(
        self,
        portfolio_value: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5,
    ) -> float:
        """
        Calculate position size using the Kelly Criterion.
        Uses a fractional Kelly (half-Kelly by default) for safety.

        Kelly formula: f* = (bp - q) / b
        where:
            b = avg_win / avg_loss (win/loss ratio)
            p = probability of winning
            q = probability of losing (1 - p)

        Args:
            portfolio_value: Current portfolio value (USD)
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            fraction: Kelly fraction multiplier (0.5 = half-Kelly for safety)

        Returns:
            Optimal position size in USD
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            logger.warning("Invalid Kelly inputs; falling back to fixed percentage")
            return self.fixed_percentage(portfolio_value)

        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1.0 - p

        kelly_pct = (b * p - q) / b

        # Clamp Kelly percentage to reasonable bounds
        kelly_pct = max(0.0, min(kelly_pct, 0.25))  # Never risk more than 25%

        # Apply fractional Kelly for safety
        kelly_pct *= fraction

        position_size = portfolio_value * kelly_pct

        logger.debug(
            f"Kelly sizing: f*={kelly_pct:.4f} (win_rate={win_rate:.2f}, "
            f"ratio={b:.2f}) => ${position_size:,.2f}"
        )
        return position_size

    def volatility_based(
        self,
        portfolio_value: float,
        atr: float,
        current_price: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on ATR (volatility-adjusted).
        Sizes the position so that a 2x ATR move equals the risk amount.

        Logic:
            risk_amount = portfolio_value * risk_pct
            shares = risk_amount / (2 * ATR)
            position_size = shares * current_price

        Args:
            portfolio_value: Current portfolio value (USD)
            atr: Current ATR value for the asset
            current_price: Current asset price
            risk_pct: Maximum risk per trade percentage

        Returns:
            Position size in USD
        """
        pct = risk_pct or self.config.max_risk_per_trade_pct
        risk_amount = portfolio_value * (pct / 100.0)

        if atr <= 0 or current_price <= 0:
            logger.warning("Invalid ATR or price; falling back to fixed percentage")
            return self.fixed_percentage(portfolio_value, pct)

        # Number of units where 2x ATR move = risk amount
        units = risk_amount / (2.0 * atr)
        position_size = units * current_price

        logger.debug(
            f"Volatility sizing: ATR={atr:.2f}, risk=${risk_amount:,.2f}, "
            f"units={units:.6f}, position=${position_size:,.2f}"
        )
        return position_size

    def calculate_position_size(
        self,
        method: str,
        portfolio_value: float,
        current_price: float,
        atr: float = 0.0,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        max_position_usd: Optional[float] = None,
    ) -> float:
        """
        Master position sizing function that dispatches to the appropriate method
        and enforces maximum position size limits.

        Args:
            method: Sizing method ("fixed", "kelly", "volatility")
            portfolio_value: Current portfolio value
            current_price: Current asset price
            atr: Current ATR (needed for volatility method)
            win_rate: Historical win rate (needed for Kelly)
            avg_win: Average win amount (needed for Kelly)
            avg_loss: Average loss amount (needed for Kelly)
            max_position_usd: Override maximum position cap

        Returns:
            Final position size in USD after applying all constraints
        """
        if method == "kelly":
            size = self.kelly_criterion(portfolio_value, win_rate, avg_win, avg_loss)
        elif method == "volatility":
            size = self.volatility_based(portfolio_value, atr, current_price)
        else:
            size = self.fixed_percentage(portfolio_value)

        # Apply maximum position limit — cap each position at 30% of portfolio
        # so multiple assets can trade simultaneously without hitting exposure limits
        max_per_position_pct = 10.0
        max_size = max_position_usd or (
            portfolio_value * max_per_position_pct / 100.0
        )
        size = min(size, max_size)

        # Ensure minimum order size
        min_size = 10.0  # Minimum $10 position
        size = max(size, min_size) if size > 0 else 0.0

        return round(size, 2)

