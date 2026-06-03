# Codex Application Summary

## Current Application

This is a Python automated trading system for BTC/USDT and XAU/USD. It includes:

- Market data feeds with fallback sources for Bitcoin and gold.
- Technical indicators, regime detection, and multiple rule-based/ML strategies.
- Ensemble signal generation.
- Portfolio, broker simulation, risk management, and backtesting.
- Flask dashboard with REST APIs for status, portfolio, risk, trades, equity, halt, and resume.
- Docker, test files, and example environment configuration.

## Reliability Improvements Made In This Copy

- Fixed short-position portfolio accounting so short equity, closing cash, and PnL are calculated consistently.
- Added mark-to-market risk value refreshes before trade approval in paper/live loops and backtests.
- Fixed backtest metrics so custom CLI balances are used by the performance analyzer.
- Changed live leverage default from 25x to 1x. Higher leverage now requires explicitly setting `LEVERAGE`.
- Added portfolio accounting tests covering both long and short positions.
- Created a clean application copy without old runtime `state_*.json` snapshots.

## Recommended Next Improvements

- Split the large `main.py` trading loops into smaller services for feeds, signal processing, execution, and lifecycle management.
- Add dependency-pinned lock files and CI so tests run consistently across machines.
- Replace broad `except Exception` blocks with typed errors and alertable failure states.
- Add data-source health scoring and prevent trading when a feed returns zero, stale, or proxy-mismatched prices.
- Persist orders, positions, and risk state in a transactional database instead of only in memory/state snapshots.
- Move the dashboard HTML/CSS/JS out of a Python string into template/static files.
- Add tests for live/paper order flipping, stop-loss/take-profit execution, and custom-balance backtest metrics.
- Add explicit paper/live kill switches, maximum leverage guards, and dry-run exchange validation before live mode starts.

