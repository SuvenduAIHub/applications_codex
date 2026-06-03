# Automated Trading System

**Production-grade automated trading platform for Bitcoin (BTC/USDT) and Gold (XAU/USD)**

A fully modular, end-to-end trading system covering data ingestion, feature engineering, multi-strategy signal generation (including ML/AI), risk management, order execution, backtesting, optimization, monitoring, and visualization.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTOMATED TRADING SYSTEM                         │
│                   BTC/USDT  &  XAU/USD Only                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  DATA LAYER  │───>│ FEATURE ENGINE   │───>│ STRATEGY LAYER   │   │
│  │              │    │                  │    │                  │   │
│  │ Binance API  │    │ RSI, MACD, EMA   │    │ Trend Following  │   │
│  │ (BTC/USDT)   │    │ Bollinger Bands  │    │ Mean Reversion   │   │
│  │              │    │ ATR, Volume      │    │ Breakout         │   │
│  │ Yahoo Finance│    │ Volatility       │    │ XGBoost ML       │   │
│  │ (XAU/USD)    │    │ Regime Detection │    │ LSTM Neural Net  │   │
│  │              │    │ BTC-Gold Corr.   │    │ Ensemble Voter   │   │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘   │
│                                                        │             │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────▼─────────┐   │
│  │  MONITORING  │<───│ EXECUTION ENGINE │<───│ RISK MANAGEMENT  │   │
│  │              │    │                  │    │                  │   │
│  │ Web Dashboard│    │ Market Orders    │    │ Kelly Criterion  │   │
│  │ REST API     │    │ Limit Orders     │    │ Fixed % Sizing   │   │
│  │ Alerts/Slack │    │ Stop Orders      │    │ Volatility-Based │   │
│  │ Trade Logs   │    │ Slippage Model   │    │ Risk Parity      │   │
│  │ Error Logs   │    │ Commission Model │    │ Max Drawdown     │   │
│  │ Recovery     │    │ Paper/Live Mode  │    │ Position Limits  │   │
│  └──────────────┘    └──────────────────┘    └──────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    BACKTESTING ENGINE                         │   │
│  │  Historical Simulation │ Walk-Forward Validation              │   │
│  │  Time-Series CV        │ Strategy Comparison                  │   │
│  │  Hyperparameter Tuning │ Performance Analytics                │   │
│  │  Equity Curve Charts   │ Drawdown Analysis                   │   │
│  │  Correlation Heatmaps  │ Trade Distribution Plots            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  METRICS: CAGR │ Sharpe │ Sortino │ Calmar │ Max DD │ Win Rate     │
│           Profit Factor │ VaR │ CVaR │ Information Ratio           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
automated-trading-system/
├── main.py                          # Application entry point & CLI
├── requirements.txt                 # Python dependencies (all open-source)
├── setup.py                         # Package setup script
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Docker Compose configuration
├── README.md                        # This file
│
├── config/
│   ├── settings.py                  # All configuration dataclasses
│   └── trading_pairs.py             # BTC/USDT and XAU/USD pair definitions
│
├── src/
│   ├── data/                        # Data ingestion layer
│   │   ├── base_feed.py             # Abstract data feed interface
│   │   ├── btc_feed.py              # Binance API for BTC/USDT
│   │   ├── gold_feed.py             # Yahoo Finance for XAU/USD
│   │   └── data_store.py            # SQLite persistence for OHLCV & trades
│   │
│   ├── indicators/                  # Feature engineering
│   │   ├── technical.py             # RSI, MACD, EMA, BB, ATR, volume features
│   │   └── regime.py                # Market regime & BTC-Gold macro detection
│   │
│   ├── strategies/                  # Trading strategies
│   │   ├── base_strategy.py         # Abstract strategy interface
│   │   ├── trend_following.py       # EMA crossover + MACD trend strategy
│   │   ├── mean_reversion.py        # Bollinger Band + RSI reversion strategy
│   │   ├── breakout.py              # Range breakout + volume confirmation
│   │   ├── ml_strategy.py           # XGBoost + LSTM (PyTorch) ML strategies
│   │   └── ensemble.py              # Adaptive regime-weighted ensemble
│   │
│   ├── execution/                   # Order execution engine
│   │   ├── order.py                 # Order data model & lifecycle
│   │   ├── portfolio.py             # Portfolio & position management
│   │   └── broker.py                # Simulated broker with slippage/fees
│   │
│   ├── risk/                        # Risk management
│   │   ├── position_sizer.py        # Kelly, fixed %, volatility-based sizing
│   │   └── risk_manager.py          # Drawdown, exposure, allocation controls
│   │
│   ├── backtesting/                 # Backtesting framework
│   │   ├── engine.py                # Historical simulation engine
│   │   ├── performance.py           # All performance metrics calculation
│   │   └── optimizer.py             # Grid search, walk-forward, CV
│   │
│   ├── monitoring/                  # Monitoring & alerting
│   │   ├── dashboard.py             # Flask web dashboard + REST API
│   │   ├── alerts.py                # Alert manager (console, file, Slack)
│   │   └── logger_config.py         # Structured logging & failure recovery
│   │
│   ├── visualization/               # Chart generation
│   │   └── charts.py                # Equity curve, drawdown, heatmap, trades
│   │
│   └── utils/
│       └── helpers.py               # Common utility functions
│
├── tests/                           # Unit tests
│   ├── test_indicators.py           # Technical indicator tests
│   ├── test_strategies.py           # Strategy signal tests
│   ├── test_risk.py                 # Risk management tests
│   └── test_backtesting.py          # Backtesting & performance tests
│
├── data/                            # Runtime data (auto-created)
├── logs/                            # Log files (auto-created)
├── reports/                         # Backtesting reports (auto-created)
│   └── charts/                      # Generated visualization charts
└── state/                           # Recovery state files (auto-created)
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Local Setup

```bash
# 1. Clone or extract the project
cd automated-trading-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run backtesting (default mode)
python main.py --mode backtest

# 5. Run backtesting with visualizations
python main.py --mode backtest --visualize

# 6. Run paper trading with live dashboard
python main.py --mode paper

# 7. Run unit tests
pytest tests/ -v
```

### Docker Setup

```bash
# Build and run backtester
docker compose --profile backtest up backtester

# Run paper trading with dashboard
docker compose up trading-system

# Access dashboard at http://localhost:5000
```

---

## Operating Modes

### 1. Backtesting Mode (`--mode backtest`)
Runs strategies against historical data with full performance analysis.

```bash
python main.py --mode backtest --balance 100000 --visualize
```

**Output:**
- Performance report (CAGR, Sharpe, Sortino, Max DD, Win Rate, etc.)
- Strategy comparison table
- Execution statistics
- Charts (equity curve, drawdown, correlation heatmap, trade analysis)

### 2. Paper Trading Mode (`--mode paper`)
Simulated live trading with real-time price feeds.

```bash
python main.py --mode paper --balance 50000 --dashboard-port 5000
```

**Features:**
- Fetches live BTC and Gold prices every 60 seconds
- Generates and executes signals in simulated environment
- Web dashboard at `http://localhost:5000`
- REST API for external control
- Automatic state recovery on restart

### 3. Live Trading Mode (`--mode live`)
Currently runs as paper trading for safety. Extend the `SimulatedBroker` class with real exchange APIs for live execution.

---

## Strategies

### Trend Following
- **Signal:** EMA crossover (12/26) with MACD confirmation
- **Filter:** RSI range (40-75 for buys), price vs SMA 50
- **Best in:** Trending markets (uptrend/downtrend regimes)
- **Stop-loss:** 2x ATR from entry

### Mean Reversion
- **Signal:** Price touching Bollinger Band extremes + RSI oversold/overbought
- **Filter:** Regime detection (skipped in strong trends)
- **Best in:** Range-bound/sideways markets
- **Target:** Middle Bollinger Band

### Breakout
- **Signal:** Price breaking above/below consolidation range
- **Confirmation:** Volume spike > 1.5x average, ATR expansion
- **Best in:** Post-consolidation, high-volatility regimes
- **Target:** 2x range height from breakout

### XGBoost ML
- **Features:** All technical indicators as input features
- **Target:** Binary classification (price up/down)
- **Regularization:** L1/L2, max_depth=5, subsample=0.8
- **Output:** Probability-based confidence scoring

### LSTM Neural Network
- **Architecture:** 2-layer LSTM (64 hidden units) + FC output
- **Input:** Sequences of 30 timesteps of indicator features
- **Training:** BCELoss, Adam optimizer, early stopping
- **Regularization:** Dropout 0.2 to prevent overfitting

### Ensemble (Default)
- **Method:** Weighted voting across all strategies
- **Adaptive:** Weights adjust based on market regime
- **Performance-tracked:** Strategies with higher recent win rates get more weight
- **Consensus:** Configurable minimum agreement threshold

---

## Risk Management

| Feature | Description |
|---------|-------------|
| **Position Sizing** | Kelly Criterion, Fixed %, Volatility-based (ATR) |
| **Stop-Loss** | ATR-based or percentage-based, per position |
| **Take-Profit** | Minimum risk-reward ratio enforced (default 1.5:1) |
| **Trailing Stop** | Activates after 4% profit, trails at 2% distance |
| **Max Drawdown** | Circuit breaker halts trading at 15% drawdown |
| **Daily Loss Limit** | Stops trading at 5% daily loss |
| **Consecutive Losses** | 60-min cooldown after 5 consecutive losses |
| **Exposure Limit** | Max 30% of portfolio in open positions |
| **Asset Allocation** | Max 60% per asset (BTC or Gold) |
| **Risk Parity** | Inverse-volatility weighting between BTC and Gold |

---

## Backtesting Metrics

The system calculates comprehensive performance metrics:

**Return Metrics:** Total Return, CAGR, Total P&L
**Risk Metrics:** Annual Volatility, VaR (95%), CVaR, Skewness, Kurtosis
**Ratios:** Sharpe, Sortino, Calmar, Information Ratio
**Drawdown:** Max Drawdown, Max DD Duration, Average Drawdown
**Trade Stats:** Win Rate, Profit Factor, Avg Win/Loss, Max Consecutive W/L

---

## Evaluation & Optimization

- **Grid Search:** Test parameter combinations and find optimal settings
- **Walk-Forward Validation:** Sliding window train/test to detect overfitting
- **Time-Series Cross-Validation:** Expanding window CV with purging
- **Strategy Comparison:** Side-by-side metrics for all strategies

---

## Monitoring Dashboard

The web dashboard provides real-time monitoring at `http://localhost:5000`:

- Portfolio value, cash, and return %
- Open positions with live PnL
- Risk exposure and drawdown
- Recent trade history
- System status and alerts

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/status` | GET | Full system status |
| `/api/portfolio` | GET | Portfolio details |
| `/api/risk` | GET | Risk metrics |
| `/api/trades` | GET | Trade history |
| `/api/equity` | GET | Equity curve data |
| `/api/halt` | POST | Emergency halt trading |
| `/api/resume` | POST | Resume after halt |

---

## Advanced Features

### BTC-Gold Macro Correlation
The system detects macro regimes by analyzing BTC-Gold correlation:
- **Risk-On:** Both assets rising, BTC outperforming
- **Risk-Off:** Gold outperforming, flight to safety
- **Divergent:** Negative correlation between assets
- **Neutral:** No clear macro signal

### Adaptive Strategy Switching
The ensemble dynamically adjusts strategy weights based on detected market regime:
- Trending → Upweight trend-following
- Ranging → Upweight mean-reversion
- High volatility → Upweight breakout
- Performance tracking → Upweight recently profitable strategies

---

## Visualization Outputs

When running with `--visualize`:

1. **Equity Curve** (`reports/charts/equity_curve.png`) - Portfolio value over time
2. **Drawdown Chart** (`reports/charts/drawdown_chart.png`) - Drawdown periods and max DD
3. **Correlation Heatmap** (`reports/charts/correlation_heatmap.png`) - BTC-Gold feature correlations
4. **Trade Analysis** (`reports/charts/trade_analysis.png`) - PnL distribution, cumulative PnL, win/loss breakdown

---

## Configuration

All parameters are configurable via `config/settings.py`:

```python
# Example: Customize strategy parameters
config = TradingSystemConfig()
config.strategy.rsi_period = 14
config.strategy.ema_fast_period = 12
config.risk.max_drawdown_pct = 10.0
config.backtest.initial_balance_usd = 200000
```

---

## Deployment Guide

### Local Deployment
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py --mode paper
```

### Docker Deployment
```bash
docker compose up -d trading-system
# Dashboard: http://localhost:5000
```

### Cloud Deployment (AWS/GCP/Azure)
1. Build Docker image: `docker build -t trading-system .`
2. Push to container registry (ECR, GCR, ACR)
3. Deploy to container service (ECS, Cloud Run, ACI)
4. Mount persistent volumes for `/app/data` and `/app/logs`
5. Configure health check on `/api/health`
6. Set environment variables for alerts (SLACK_WEBHOOK_URL)

### Scaling Suggestions
- **Horizontal:** Run separate instances per trading pair
- **Database:** Migrate from SQLite to PostgreSQL for concurrent access
- **Message Queue:** Add RabbitMQ/Redis for async signal processing
- **Caching:** Add Redis for real-time data caching
- **Monitoring:** Integrate Prometheus + Grafana for metrics
- **CI/CD:** GitHub Actions for automated testing and deployment

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_indicators.py -v
```

---

## Technology Stack

All open-source:

| Category | Technologies |
|----------|-------------|
| Language | Python 3.10+ |
| Data | pandas, numpy, scipy |
| ML/AI | scikit-learn, XGBoost, PyTorch |
| Indicators | ta (Technical Analysis library) |
| Web | Flask, Flask-SocketIO, Flask-CORS |
| Visualization | matplotlib, seaborn |
| Database | SQLite (via SQLAlchemy ORM) |
| Logging | loguru, rich |
| Testing | pytest, pytest-cov |
| Container | Docker, Docker Compose |
| APIs | Binance (public), Yahoo Finance (public) |

---

## Constraints & Notes

- **BTC/USDT and XAU/USD only** — no other pairs
- **Anti-overfitting:** Walk-forward validation, regularization in ML models, early stopping
- **Reproducibility:** Fixed random seeds (42) in all ML training
- **No paid APIs:** All data sources are free/public
- **Production-ready:** Structured logging, error recovery, circuit breakers, health checks
