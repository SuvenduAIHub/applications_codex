"""
Global configuration settings for the Automated Trading System.
Defines all configurable parameters for data feeds, strategies,
risk management, execution, monitoring, and multi-currency support.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TradingMode(Enum):
    """Operating modes for the trading system."""
    LIVE = "live"            # Real-time trading with live execution
    PAPER = "paper"          # Paper trading with simulated execution
    BACKTEST = "backtest"    # Historical backtesting mode


class BaseCurrency(Enum):
    """Supported base currencies for deposits and portfolio display."""
    USD = "USD"     # US Dollar / USDT (Binance)
    INR = "INR"     # Indian Rupee (WazirX)


class OrderSide(Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Lifecycle status of an order."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class TimeFrame(Enum):
    """Supported candlestick timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class DataFeedConfig:
    """Configuration for market data feeds."""
    # Binance API for BTC/USDT (public, no API key required for market data)
    binance_base_url: str = "https://api.binance.com"
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"

    # Gold price feed using open-source Yahoo Finance scraping endpoint
    gold_api_url: str = "https://query1.finance.yahoo.com/v8/finance/chart"

    # Data polling interval in seconds for REST-based feeds (configurable via SCAN_INTERVAL_SECONDS env var)
    polling_interval_seconds: int = field(default_factory=lambda: int(os.environ.get("SCAN_INTERVAL_SECONDS", "15")))

    # Maximum number of historical candles to fetch per request
    max_candles_per_request: int = 1000

    # Default timeframe for candle data
    default_timeframe: TimeFrame = TimeFrame.H1

    # Local cache directory for storing downloaded historical data
    cache_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")

    # Number of retry attempts for failed API calls
    api_retry_count: int = 3

    # Timeout in seconds for API requests
    api_timeout_seconds: int = 30


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    # --- Trend Following (EMA Crossover) ---
    ema_fast_period: int = 12        # Fast EMA period
    ema_slow_period: int = 26        # Slow EMA period
    ema_signal_period: int = 9       # Signal line period (MACD-like)

    # --- RSI Mean Reversion ---
    rsi_period: int = 14             # RSI calculation period
    rsi_overbought: float = 70.0     # RSI overbought threshold (sell signal) — aggressive for more trades
    rsi_oversold: float = 30.0       # RSI oversold threshold (buy signal) — aggressive for more trades

    # --- Bollinger Bands ---
    bb_period: int = 20              # Bollinger Band period
    bb_std_dev: float = 2.0          # Standard deviation multiplier

    # --- MACD ---
    macd_fast: int = 12              # MACD fast EMA period
    macd_slow: int = 26              # MACD slow EMA period
    macd_signal: int = 9             # MACD signal line period

    # --- ATR (Average True Range) for volatility ---
    atr_period: int = 14             # ATR calculation period

    # --- Volume analysis ---
    volume_ma_period: int = 20       # Volume moving average period
    volume_spike_multiplier: float = 1.5  # Volume spike detection threshold

    # Minimum number of confirming indicators to trigger a trade
    # Requires 2 strategies to agree — reduces whipsaw from weak/noisy signals
    min_signal_confirmations: int = 3

    # Cooldown period between trades (in candles)
    # 3 candles × 5 min = 15 min cooldown — prevents rapid flip-flopping
    trade_cooldown_candles: int = 6


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Maximum percentage of portfolio to risk per trade
    max_risk_per_trade_pct: float = 0.75

    # Maximum total portfolio exposure percentage — high for active scalping
    max_portfolio_exposure_pct: float = field(
        default_factory=lambda: float(os.environ.get("MAX_PORTFOLIO_EXPOSURE_PCT", "35.0"))
    )

    # Maximum number of concurrent open positions
    max_concurrent_positions: int = 2

    # Maximum drawdown before system halts trading (percentage)
    max_drawdown_pct: float = 15.0

    # Stop-loss percentage from entry price — wider for bigger profit targets
    default_stop_loss_pct: float = field(
        default_factory=lambda: float(os.environ.get("DEFAULT_STOP_LOSS_PCT", "2.0"))
    )

    # Optional fixed price-distance stop loss in USD.
    # Example: 200 means long SL = entry - 200, short SL = entry + 200.
    fixed_stop_loss_usd: float = field(
        default_factory=lambda: float(os.environ.get("FIXED_STOP_LOSS_USD", "0"))
    )

    # Take-profit percentage from entry price — targets meaningful moves
    default_take_profit_pct: float = 4.0

    # Optional fixed price-distance take profit in USD.
    # Set to 0 to let trades run until stop/trailing stop when dollar risk controls are enabled.
    fixed_take_profit_usd: float = field(
        default_factory=lambda: float(os.environ.get("FIXED_TAKE_PROFIT_USD", "0"))
    )

    # Trailing stop activation percentage — locks in profit after 3% move
    trailing_stop_activation_pct: float = field(
        default_factory=lambda: float(os.environ.get("TRAILING_STOP_ACTIVATION_PCT", "3.0"))
    )

    # Trailing stop distance percentage — trails 1.5% behind the peak
    trailing_stop_distance_pct: float = field(
        default_factory=lambda: float(os.environ.get("TRAILING_STOP_DISTANCE_PCT", "1.5"))
    )

    # Optional fixed trailing stop in USD.
    # Example: activation=100, distance=20 means trail starts after $100 profit
    # and keeps the stop $20 behind the best price.
    trailing_stop_activation_usd: float = field(
        default_factory=lambda: float(os.environ.get("TRAILING_STOP_ACTIVATION_USD", "0"))
    )
    trailing_stop_distance_usd: float = field(
        default_factory=lambda: float(os.environ.get("TRAILING_STOP_DISTANCE_USD", "0"))
    )

    # Risk-reward ratio minimum threshold
    min_risk_reward_ratio: float = 1.5

    # Maximum daily loss limit (percentage of portfolio)
    max_daily_loss_pct: float = 2.0

    # Maximum number of losing trades before cooldown
    max_consecutive_losses: int = 3

    # Cooldown period after max consecutive losses (in minutes)
    loss_streak_cooldown_minutes: int = 120

    # Per-asset allocation limits (percentage of total portfolio) — high for active scalping
    asset_allocation_limits: Dict[str, float] = field(default_factory=lambda: {
        "BTC/USDT": 40.0,
        "XAU/USD": 40.0,
    })


@dataclass
class ExecutionConfig:
    """Configuration for order execution."""
    # Simulated slippage percentage for paper/backtest modes
    simulated_slippage_pct: float = 0.1

    # Simulated commission/fee percentage per trade
    simulated_commission_pct: float = 0.1

    # Maximum time for a limit order to remain active (seconds)
    limit_order_timeout_seconds: int = 3600

    # Minimum order size in USD equivalent
    min_order_size_usd: float = 10.0

    # Maximum order size in USD equivalent
    max_order_size_usd: float = 100000.0

    # Order fill simulation delay in seconds (for paper trading)
    fill_simulation_delay_seconds: float = 0.5

    # --- Live Exchange API Credentials (loaded from environment) ---
    # Binance API (for USDT-based trading)
    binance_api_key: str = field(default_factory=lambda: os.environ.get("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.environ.get("BINANCE_API_SECRET", ""))
    binance_testnet: bool = field(default_factory=lambda: os.environ.get("BINANCE_TESTNET", "true").lower() == "true")

    # WazirX API (for INR-based trading)
    wazirx_api_key: str = field(default_factory=lambda: os.environ.get("WAZIRX_API_KEY", ""))
    wazirx_api_secret: str = field(default_factory=lambda: os.environ.get("WAZIRX_API_SECRET", ""))

    # Delta Exchange API (for INR + USDT trading, India-based derivatives + spot)
    delta_api_key: str = field(default_factory=lambda: os.environ.get("DELTA_API_KEY", ""))
    delta_api_secret: str = field(default_factory=lambda: os.environ.get("DELTA_API_SECRET", ""))
    delta_testnet: bool = field(default_factory=lambda: os.environ.get("DELTA_TESTNET", "true").lower() == "true")

    # XM (MetaTrader 5) API — supports Gold (XAU/USD) and BTC/USD CFD trading from India
    xm_mt5_login: str = field(default_factory=lambda: os.environ.get("XM_MT5_LOGIN", ""))
    xm_mt5_password: str = field(default_factory=lambda: os.environ.get("XM_MT5_PASSWORD", ""))
    xm_mt5_server: str = field(default_factory=lambda: os.environ.get("XM_MT5_SERVER", "XMGlobal-MT5"))
    xm_demo: bool = field(default_factory=lambda: os.environ.get("XM_DEMO", "true").lower() == "true")

    # Leverage for live trading. Keep default at 1x; higher leverage must be explicit.
    leverage: int = field(default_factory=lambda: int(os.environ.get("LEVERAGE", "1")))


@dataclass
class CurrencyConfig:
    """Configuration for multi-currency support (INR + USD)."""
    # User's selected base currency for deposits and display
    base_currency: BaseCurrency = field(
        default_factory=lambda: BaseCurrency(os.environ.get("BASE_CURRENCY", "USD"))
    )

    # Exchange rate API cache duration in seconds
    rate_cache_ttl_seconds: int = 300

    # Fallback INR/USD rate if API is unavailable
    fallback_inr_per_usd: float = 83.50


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and alerting."""
    # Flask dashboard host and port
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 5000

    # Enable/disable the web dashboard
    dashboard_enabled: bool = True

    # Log file path
    log_file: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "trading.log")

    # Log rotation size in MB
    log_rotation_mb: int = 50

    # Number of log files to retain
    log_retention_count: int = 10

    # Metrics collection interval (seconds)
    metrics_interval_seconds: int = 30

    # Alert thresholds
    alert_on_drawdown_pct: float = 10.0      # Alert when drawdown exceeds this
    alert_on_daily_loss_pct: float = 3.0      # Alert when daily loss exceeds this
    alert_on_position_size_pct: float = 25.0  # Alert when single position exceeds this


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""
    # Default historical period for backtesting (days)
    default_lookback_days: int = 365

    # Initial portfolio balance for backtesting (USD)
    initial_balance_usd: float = 100000.0

    # Include transaction costs in backtest
    include_transaction_costs: bool = True

    # Include slippage simulation in backtest
    include_slippage: bool = True

    # Report output directory
    report_output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

    # Walk-forward optimization window size (days)
    walk_forward_window_days: int = 90

    # Walk-forward step size (days)
    walk_forward_step_days: int = 30


@dataclass
class TradingSystemConfig:
    """Master configuration aggregating all sub-configurations."""
    # Operating mode (live, paper, backtest)
    mode: TradingMode = TradingMode.PAPER

    # Trading pairs to trade
    trading_pairs: List[str] = field(default_factory=lambda: ["BTC/USDT", "XAU/USD"])

    # Sub-configurations
    data_feed: DataFeedConfig = field(default_factory=DataFeedConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    currency: CurrencyConfig = field(default_factory=CurrencyConfig)

    # Database path for trade/order persistence
    database_url: str = field(default_factory=lambda: "sqlite:///" + os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "trading.db"
    ))


def load_config() -> TradingSystemConfig:
    """
    Load and return the default trading system configuration.
    Reads BASE_CURRENCY and exchange API keys from environment variables.
    """
    return TradingSystemConfig()


