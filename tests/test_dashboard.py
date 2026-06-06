"""
Unit tests for the monitoring dashboard REST API.
Tests that exchange name, currency, and dual-currency data are
correctly exposed in API responses.
"""

import json
import pytest

from src.monitoring.dashboard import TradingDashboard
from src.execution.portfolio import Portfolio
from src.risk.risk_manager import RiskManager
from config.settings import RiskConfig


@pytest.fixture
def dashboard_app():
    """Create a test Flask app from TradingDashboard."""
    dashboard = TradingDashboard(port=5099)
    portfolio = Portfolio(initial_balance=100000.0)
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    risk_manager.initialize(100000.0)

    dashboard.set_components(
        portfolio=portfolio,
        risk_manager=risk_manager,
        strategy=None,
        mode="live",
        exchange_name="binance",
        base_currency="USD",
    )
    # Return the Flask test client for API testing
    return dashboard.app.test_client()


@pytest.fixture
def paper_dashboard_app():
    """Create a dashboard app in paper mode with no exchange."""
    dashboard = TradingDashboard(port=5098)
    portfolio = Portfolio(initial_balance=100000.0)
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    risk_manager.initialize(100000.0)

    dashboard.set_components(
        portfolio=portfolio,
        risk_manager=risk_manager,
        strategy=None,
        mode="paper",
        exchange_name="paper",
        base_currency="INR",
    )
    return dashboard.app.test_client()


class TestDashboardStatusAPI:
    """Tests for /api/status endpoint exchange and currency fields."""

    def test_status_contains_exchange(self, dashboard_app):
        """Status response should include the active exchange name."""
        response = dashboard_app.get("/api/status")
        data = json.loads(response.data)
        assert data["system"]["exchange"] == "binance"

    def test_status_contains_mode(self, dashboard_app):
        """Status response should include the trading mode."""
        response = dashboard_app.get("/api/status")
        data = json.loads(response.data)
        assert data["system"]["mode"] == "live"

    def test_status_contains_base_currency(self, dashboard_app):
        """Status response should include the base currency."""
        response = dashboard_app.get("/api/status")
        data = json.loads(response.data)
        assert data["system"]["base_currency"] == "USD"

    def test_status_currency_block(self, dashboard_app):
        """Status should include a currency block with base and rate."""
        response = dashboard_app.get("/api/status")
        data = json.loads(response.data)
        assert "currency" in data
        assert data["currency"]["base"] == "USD"
        assert "inr_usd_rate" in data["currency"]

    def test_status_portfolio_data(self, dashboard_app):
        """Status should include portfolio data."""
        response = dashboard_app.get("/api/status")
        data = json.loads(response.data)
        assert "portfolio" in data
        assert data["portfolio"]["total_value"] == 100000.0

    def test_paper_mode_exchange_name(self, paper_dashboard_app):
        """Paper mode should show 'paper' as exchange name."""
        response = paper_dashboard_app.get("/api/status")
        data = json.loads(response.data)
        assert data["system"]["exchange"] == "paper"
        assert data["system"]["mode"] == "paper"

    def test_paper_mode_inr_currency(self, paper_dashboard_app):
        """Paper mode with INR should show INR as base currency."""
        response = paper_dashboard_app.get("/api/status")
        data = json.loads(response.data)
        assert data["system"]["base_currency"] == "INR"
        assert data["currency"]["base"] == "INR"


class TestDashboardHealthAPI:
    """Tests for /api/health endpoint."""

    def test_health_returns_ok(self, dashboard_app):
        """Health endpoint should return healthy status."""
        response = dashboard_app.get("/api/health")
        data = json.loads(response.data)
        assert data["status"] == "healthy"

    def test_health_contains_exchange(self, dashboard_app):
        """Health endpoint should include the exchange name."""
        response = dashboard_app.get("/api/health")
        data = json.loads(response.data)
        assert data["exchange"] == "binance"

    def test_health_contains_mode(self, dashboard_app):
        """Health endpoint should include the trading mode."""
        response = dashboard_app.get("/api/health")
        data = json.loads(response.data)
        assert data["mode"] == "live"

    def test_health_contains_currency(self, dashboard_app):
        """Health endpoint should include the base currency."""
        response = dashboard_app.get("/api/health")
        data = json.loads(response.data)
        assert data["base_currency"] == "USD"


class TestDashboardPortfolioAPI:
    """Tests for /api/portfolio endpoint."""

    def test_portfolio_returns_data(self, dashboard_app):
        """Portfolio endpoint should return portfolio summary."""
        response = dashboard_app.get("/api/portfolio")
        data = json.loads(response.data)
        assert data["total_value"] == 100000.0
        assert data["cash_balance"] == 100000.0
        assert data["currency"] == "USD"

    def test_portfolio_trades_count(self, dashboard_app):
        """Portfolio should show zero trades initially."""
        response = dashboard_app.get("/api/portfolio")
        data = json.loads(response.data)
        assert data["total_trades"] == 0


class TestDashboardCurrencyAPI:
    """Tests for /api/currency endpoint."""

    def test_currency_endpoint_exists(self, dashboard_app):
        """Currency endpoint should return 200."""
        response = dashboard_app.get("/api/currency")
        assert response.status_code == 200

    def test_currency_returns_supported_list(self, dashboard_app):
        """Currency endpoint should list supported currencies."""
        response = dashboard_app.get("/api/currency")
        data = json.loads(response.data)
        assert "supported_currencies" in data
        assert "USD" in data["supported_currencies"]
        assert "INR" in data["supported_currencies"]

    def test_currency_returns_base(self, dashboard_app):
        """Currency endpoint should return active base currency."""
        response = dashboard_app.get("/api/currency")
        data = json.loads(response.data)
        assert data["base_currency"] == "USD"


class TestDashboardTradesAPI:
    """Tests for /api/trades and /api/equity endpoints."""

    def test_trades_empty_initially(self, dashboard_app):
        """Trades endpoint should return empty list initially."""
        response = dashboard_app.get("/api/trades")
        data = json.loads(response.data)
        assert data == []

    def test_equity_empty_initially(self, dashboard_app):
        """Equity endpoint should return empty list initially."""
        response = dashboard_app.get("/api/equity")
        data = json.loads(response.data)
        assert data == []


class TestDashboardControlAPI:
    """Tests for /api/halt and /api/resume control endpoints."""

    def test_halt_endpoint(self, dashboard_app):
        """Halt endpoint should return acknowledgement."""
        response = dashboard_app.post("/api/halt")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "halt" in data.get("status", "").lower() or "halt" in str(data).lower()

    def test_resume_endpoint(self, dashboard_app):
        """Resume endpoint should return acknowledgement."""
        response = dashboard_app.post("/api/resume")
        assert response.status_code == 200


class TestDashboardLogin:
    """Tests for optional dashboard form authentication."""

    def test_login_redirects_dashboard_when_enabled(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_USERNAME", "test-user")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pass")
        dashboard = TradingDashboard(port=5097)
        client = dashboard.app.test_client()

        response = client.get("/")

        assert response.status_code == 302
        assert "/login" in response.headers["Location"]

    def test_login_allows_dashboard_after_valid_credentials(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_USERNAME", "test-user")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pass")
        dashboard = TradingDashboard(port=5096)
        client = dashboard.app.test_client()

        response = client.post(
            "/login",
            data={"username": "test-user", "password": "test-pass"},
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert b"Suvshrabani AI Trading System" in response.data

    def test_health_stays_public_for_container_healthcheck(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_USERNAME", "test-user")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pass")
        dashboard = TradingDashboard(port=5095)
        client = dashboard.app.test_client()

        response = client.get("/api/health")

        assert response.status_code == 200
