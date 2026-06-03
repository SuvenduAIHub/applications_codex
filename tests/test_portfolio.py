"""
Tests for portfolio accounting.
"""

import pytest

from src.execution.portfolio import Portfolio


def test_long_position_equity_updates_with_price():
    portfolio = Portfolio(initial_balance=100000.0)

    portfolio.open_position("BTC/USDT", "buy", quantity=1.0, price=50000.0)
    assert portfolio.cash_balance == pytest.approx(50000.0)
    assert portfolio.total_value == pytest.approx(100000.0)

    portfolio.update_prices({"BTC/USDT": 55000.0})
    assert portfolio.total_value == pytest.approx(105000.0)

    result = portfolio.close_position("BTC/USDT", price=55000.0)
    assert result["pnl"] == pytest.approx(5000.0)
    assert portfolio.cash_balance == pytest.approx(105000.0)


def test_short_position_equity_updates_with_price():
    portfolio = Portfolio(initial_balance=100000.0)

    portfolio.open_position("BTC/USDT", "sell", quantity=1.0, price=50000.0)
    assert portfolio.cash_balance == pytest.approx(150000.0)
    assert portfolio.total_value == pytest.approx(100000.0)

    portfolio.update_prices({"BTC/USDT": 45000.0})
    assert portfolio.total_value == pytest.approx(105000.0)

    result = portfolio.close_position("BTC/USDT", price=45000.0)
    assert result["pnl"] == pytest.approx(5000.0)
    assert portfolio.cash_balance == pytest.approx(105000.0)

