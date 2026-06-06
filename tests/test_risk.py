"""
Unit tests for the risk management module.
Tests position sizing, risk constraints, stop-loss, and portfolio limits.
"""

import pytest

from config.settings import RiskConfig
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager


class TestPositionSizer:
    """Tests for position sizing calculations."""

    def test_fixed_percentage(self):
        """Fixed percentage sizing should return correct amount."""
        sizer = PositionSizer()
        size = sizer.fixed_percentage(100000, 2.0)
        assert size == pytest.approx(2000.0)

    def test_fixed_percentage_default(self):
        """Should use config default when no percentage specified."""
        config = RiskConfig(max_risk_per_trade_pct=3.0)
        sizer = PositionSizer(config)
        size = sizer.fixed_percentage(100000)
        assert size == pytest.approx(3000.0)

    def test_kelly_criterion_positive(self):
        """Kelly should return positive size for profitable strategy."""
        sizer = PositionSizer()
        size = sizer.kelly_criterion(
            portfolio_value=100000,
            win_rate=0.6,
            avg_win=200,
            avg_loss=100,
        )
        assert size > 0

    def test_kelly_criterion_capped(self):
        """Kelly should never exceed 25% of portfolio."""
        sizer = PositionSizer()
        size = sizer.kelly_criterion(
            portfolio_value=100000,
            win_rate=0.9,
            avg_win=1000,
            avg_loss=10,
        )
        assert size <= 25000  # 25% cap

    def test_volatility_based(self):
        """Volatility-based sizing should return positive size."""
        sizer = PositionSizer()
        size = sizer.volatility_based(
            portfolio_value=100000,
            atr=500,
            current_price=50000,
        )
        assert size > 0

    def test_volatility_based_lower_with_higher_atr(self):
        """Higher ATR (more volatile) should produce smaller position."""
        sizer = PositionSizer()
        size_low_vol = sizer.volatility_based(100000, atr=100, current_price=50000)
        size_high_vol = sizer.volatility_based(100000, atr=500, current_price=50000)
        assert size_low_vol > size_high_vol

    def test_calculate_position_size_respects_max(self):
        """Master sizing function should enforce maximum limits."""
        sizer = PositionSizer()
        size = sizer.calculate_position_size(
            method="fixed",
            portfolio_value=100000,
            current_price=50000,
            max_position_usd=500,
        )
        assert size <= 500


class TestRiskManager:
    """Tests for the risk manager."""

    def setup_method(self):
        """Set up a fresh risk manager for each test."""
        self.config = RiskConfig(
            max_risk_per_trade_pct=2.0,
            max_portfolio_exposure_pct=30.0,
            max_concurrent_positions=4,
            max_drawdown_pct=15.0,
            max_daily_loss_pct=5.0,
            max_consecutive_losses=5,
        )
        self.rm = RiskManager(self.config)
        self.rm.initialize(100000.0)

    def test_can_trade_basic(self):
        """Should approve a valid trade."""
        allowed, reason = self.rm.can_trade("BTC/USDT", "buy", 5000)
        assert allowed
        assert reason == "Trade approved"

    def test_blocks_when_halted(self):
        """Should reject trades when trading is halted."""
        self.rm.trading_halted = True
        self.rm.halt_reason = "Test halt"
        allowed, reason = self.rm.can_trade("BTC/USDT", "buy", 1000)
        assert not allowed
        assert "halted" in reason.lower()

    def test_blocks_exceeding_exposure(self):
        """Should reject trades that exceed portfolio exposure limit."""
        # Max exposure is 30% of 100k = 30k
        self.rm.register_position("BTC/USDT", "buy", 25000, 50000, 48000, 55000)
        allowed, reason = self.rm.can_trade("XAU/USD", "buy", 10000)
        assert not allowed
        assert "exposure" in reason.lower()

    def test_blocks_max_positions(self):
        """Should reject trades when max concurrent positions reached."""
        for i in range(4):
            self.rm.register_position(f"PAIR{i}", "buy", 1000, 100, 95, 110)
        allowed, reason = self.rm.can_trade("NEW/PAIR", "buy", 1000)
        assert not allowed
        assert "positions" in reason.lower()

    def test_stop_loss_calculation(self):
        """Stop-loss should be below entry for buys, above for sells."""
        sl_buy = self.rm.calculate_stop_loss(50000, "buy", atr=1000)
        assert sl_buy < 50000

        sl_sell = self.rm.calculate_stop_loss(50000, "sell", atr=1000)
        assert sl_sell > 50000

    def test_take_profit_calculation(self):
        """Take-profit should respect minimum risk-reward ratio."""
        sl = self.rm.calculate_stop_loss(50000, "buy", atr=1000)
        tp = self.rm.calculate_take_profit(50000, "buy", sl)

        risk = 50000 - sl
        reward = tp - 50000
        # Reward should be at least min_risk_reward_ratio * risk
        assert reward >= risk * self.config.min_risk_reward_ratio

    def test_close_position_tracks_pnl(self):
        """Closing a position should correctly track PnL."""
        self.rm.register_position("BTC/USDT", "buy", 10000, 50000, 48000, 55000)
        result = self.rm.close_position("BTC/USDT", 52000)
        assert result is not None
        assert result["pnl_usd"] > 0  # Price went up, so profit

    def test_consecutive_losses_tracking(self):
        """Should track consecutive losses correctly."""
        for _ in range(3):
            self.rm.register_position("BTC/USDT", "buy", 1000, 50000, 48000, 55000)
            self.rm.close_position("BTC/USDT", 48000)  # Loss
        assert self.rm.consecutive_losses == 3

    def test_risk_parity_allocation(self):
        """Risk parity should allocate more to less volatile asset."""
        # BTC is more volatile than Gold
        alloc = self.rm.get_portfolio_risk_parity_allocation(
            btc_volatility=0.60,  # 60% annual vol
            gold_volatility=0.15,  # 15% annual vol
        )
        assert alloc["XAU/USD"] > alloc["BTC/USDT"]  # Gold gets more weight
        # Allocations should not exceed limits
        assert alloc["BTC/USDT"] <= self.config.asset_allocation_limits.get("BTC/USDT", 100)
        assert alloc["XAU/USD"] <= self.config.asset_allocation_limits.get("XAU/USD", 100)

    def test_check_stop_levels(self):
        """Should detect when price hits stop-loss or take-profit."""
        self.rm.register_position("BTC/USDT", "buy", 10000, 50000, 48000, 55000)
        # Price above entry but below TP
        assert self.rm.check_stop_levels("BTC/USDT", 52000) is None
        # Price hits stop-loss
        assert self.rm.check_stop_levels("BTC/USDT", 47000) == "stop_loss"
        # Reset position and test take-profit
        self.rm.register_position("BTC/USDT", "buy", 10000, 50000, 48000, 55000)
        assert self.rm.check_stop_levels("BTC/USDT", 56000) == "take_profit"

    def test_fixed_dollar_stop_uses_position_pnl_not_raw_price_distance(self):
        """Fixed dollar stop should represent account PnL on the leveraged notional."""
        config = RiskConfig(fixed_stop_loss_usd=200)
        rm = RiskManager(config)
        entry = 60000.0
        margin = 500.0
        notional = margin * 25

        stop_loss = rm.calculate_stop_loss(
            entry,
            "buy",
            position_notional_usd=notional,
        )

        assert stop_loss == pytest.approx(59040.0)

    def test_trailing_stop_uses_position_pnl_dollars(self):
        """Dollar trailing should activate on actual leveraged PnL and trail by PnL dollars."""
        config = RiskConfig(
            fixed_stop_loss_usd=200,
            trailing_stop_activation_usd=100,
            trailing_stop_distance_usd=20,
        )
        rm = RiskManager(config)
        rm.initialize(1000.0)
        entry = 60000.0
        margin = 500.0
        notional = margin * 25
        stop_loss = rm.calculate_stop_loss(entry, "buy", position_notional_usd=notional)
        rm.register_position(
            "BTC/USDT",
            "buy",
            margin,
            entry,
            stop_loss,
            None,
            notional_usd=notional,
        )

        assert rm.update_trailing_stop("BTC/USDT", 60400.0) is None
        new_stop = rm.update_trailing_stop("BTC/USDT", 60480.0)

        assert new_stop == pytest.approx(60384.0)

    def test_risk_summary(self):
        """Risk summary should contain expected keys."""
        summary = self.rm.get_risk_summary()
        assert "portfolio_value" in summary
        assert "drawdown_pct" in summary
        assert "daily_pnl" in summary
        assert "trading_halted" in summary

    def test_risk_summary_includes_configured_exposure_limit(self):
        """Risk summary should expose current usage and configured max exposure."""
        summary = self.rm.get_risk_summary()
        assert summary["total_exposure_usd"] == 0
        assert summary["max_exposure_pct"] == 30.0
        assert summary["max_exposure_usd"] == pytest.approx(30000.0)
        assert summary["exposure_used_pct"] == 0
