"""
Real-time monitoring dashboard and REST API.
Provides a Flask-based web dashboard for monitoring trading activity,
PnL, open positions, risk exposure, and system health.
Also exposes a REST API for external control and integration.
Uses TradingView Lightweight Charts (open-source) for interactive candlestick charting.
"""

import json
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for
from flask_cors import CORS
from flask_socketio import SocketIO
from loguru import logger

# HTML template for the monitoring dashboard — includes TradingView Lightweight Charts
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Suvshrabani AI Trading System</title>
    <!-- TradingView Lightweight Charts — open-source (Apache 2.0 license) -->
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        /* ===== CSS Variables — theme changes based on trading mode ===== */
        /* Paper/Demo: Light blue | Live: Light green background with red/green action buttons */
        :root {
            --bg-primary: {{ '#f0fdf4' if mode == 'live' else '#eef2f7' }};
            --bg-header: {{ '#f7fef9' if mode == 'live' else '#ffffff' }};
            --bg-card: #ffffff;
            --bg-chart: #ffffff;
            --border-color: {{ '#bbf7d0' if mode == 'live' else '#c7d2e0' }};
            --border-light: {{ '#dcfce7' if mode == 'live' else '#e2e8f0' }};
            --text-primary: {{ '#14532d' if mode == 'live' else '#1e293b' }};
            --text-secondary: {{ '#166534' if mode == 'live' else '#64748b' }};
            --text-heading: {{ '#15803d' if mode == 'live' else '#2563eb' }};
            --btn-bg: {{ '#dc2626' if mode == 'live' else '#3b82f6' }};
            --btn-hover: {{ '#b91c1c' if mode == 'live' else '#1d4ed8' }};
            --th-bg: {{ '#dcfce7' if mode == 'live' else '#dbeafe' }};
            --chart-grid: {{ '#dcfce7' if mode == 'live' else '#e2e8f0' }};
            --chart-text: {{ '#14532d' if mode == 'live' else '#1e293b' }};
            --chart-border: {{ '#bbf7d0' if mode == 'live' else '#cbd5e1' }};
            --mode-accent: {{ '#dc2626' if mode == 'live' else '#3b82f6' }};
        }
        /* Dark theme — applied via body.dark class */
        body.dark {
            --bg-primary: {{ '#052e16' if mode == 'live' else '#0f172a' }};
            --bg-header: {{ '#14532d' if mode == 'live' else '#1e293b' }};
            --bg-card: {{ '#14532d' if mode == 'live' else '#1e293b' }};
            --bg-chart: {{ '#031a0b' if mode == 'live' else '#131722' }};
            --border-color: {{ '#166534' if mode == 'live' else '#334155' }};
            --border-light: {{ '#166534' if mode == 'live' else '#334155' }};
            --text-primary: {{ '#dcfce7' if mode == 'live' else '#e2e8f0' }};
            --text-secondary: {{ '#86efac' if mode == 'live' else '#94a3b8' }};
            --text-heading: {{ '#4ade80' if mode == 'live' else '#60a5fa' }};
            --btn-bg: {{ '#dc2626' if mode == 'live' else '#2563eb' }};
            --btn-hover: {{ '#b91c1c' if mode == 'live' else '#1d4ed8' }};
            --th-bg: {{ '#14532d' if mode == 'live' else '#334155' }};
            --chart-grid: {{ '#052e16' if mode == 'live' else '#1e222d' }};
            --chart-text: {{ '#bbf7d0' if mode == 'live' else '#d1d4dc' }};
            --chart-border: {{ '#166534' if mode == 'live' else '#2B2B43' }};
            --mode-accent: {{ '#dc2626' if mode == 'live' else '#2563eb' }};
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg-primary); color: var(--text-primary); transition: background 0.3s, color 0.3s; font-size: 15px; font-weight: 500; }
        .header { background: var(--bg-header); padding: 15px 20px; text-align: center; border-bottom: 2px solid var(--border-color); display: flex; justify-content: center; align-items: center; position: relative; }
        .header h1 { color: var(--text-heading); font-size: 26px; font-weight: 700; }
        .header .subtitle { color: var(--text-secondary); font-size: 15px; margin-top: 4px; font-weight: 500; }
        .header-content { text-align: center; }
        .container { max-width: 1400px; margin: 0 auto; padding: 15px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin-bottom: 15px; }
        .card { background: var(--bg-card); border-radius: 10px; padding: 15px; border: 1px solid var(--border-color); transition: background 0.3s; }
        .card h2 { color: var(--text-heading); font-size: 17px; font-weight: 700; margin-bottom: 12px; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; }
        .metric { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid var(--border-light); font-size: 15px; }
        .metric .label { color: var(--text-secondary); font-weight: 500; }
        .metric .value { font-weight: 700; font-size: 15px; }
        .positive { color: #16a34a; }
        .negative { color: #dc2626; }
        body.dark .positive { color: #4ade80; }
        body.dark .negative { color: #f87171; }
        .neutral { color: var(--text-secondary); }
        table { width: 100%; border-collapse: collapse; }
        th { background: var(--th-bg); padding: 10px; text-align: left; font-size: 14px; font-weight: 700; color: var(--text-primary); }
        td { padding: 10px; border-bottom: 1px solid var(--border-light); font-size: 14px; font-weight: 500; }
        .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
        .status-active { background: #00d4aa; }
        .status-halted { background: #e94560; }

        /* Live mode indicator badge — shows LIVE or DEMO in header */
        .mode-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 700; letter-spacing: 0.5px; margin-left: 10px; vertical-align: middle; }
        .mode-badge.live { background: #dcfce7; color: #14532d; border: 1.5px solid #16a34a; animation: pulse-live 2s infinite; }
        .mode-badge.demo { background: #dbeafe; color: #1e40af; border: 1.5px solid #3b82f6; }
        @keyframes pulse-live { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
        .btn { background: var(--btn-bg); color: #fff; border: none; padding: 9px 18px; border-radius: 8px; cursor: pointer; font-size: 14px; transition: all 0.2s; box-shadow: 0 2px 4px rgba(59,130,246,0.15); font-weight: 600; }
        .btn:hover { background: var(--btn-hover); transform: translateY(-1px); box-shadow: 0 4px 8px rgba(59,130,246,0.25); }
        .btn:active { transform: translateY(0); }
        .btn.active { background: var(--btn-hover); }
        .btn-sm { padding: 6px 14px; font-size: 13px; border-radius: 6px; font-weight: 600; }
        .btn-outline { background: transparent; color: var(--btn-bg); border: 1.5px solid var(--btn-bg); box-shadow: none; }
        .btn-outline:hover { background: var(--btn-bg); color: #fff; }
        .btn-danger { background: #dc2626; }
        .btn-danger:hover { background: #b91c1c; }
        .btn-success { background: #16a34a; }
        .btn-success:hover { background: #15803d; }
        .action-bar { display: grid; grid-template-columns: auto minmax(150px, 1fr) minmax(150px, 1fr) repeat(3, auto) repeat(5, minmax(78px, auto)) auto; gap: 7px; align-items: center; margin-bottom: 10px; }
        .price-pill { border-radius: 8px; padding: 9px 12px; font-weight: 800; text-align: center; white-space: nowrap; font-size: 13px; }
        .price-pill.btc { background: #f97316; color: #fff; }
        .price-pill.gold { background: #facc15; color: #111827; }
        .price-change { display: inline-block; background: #047857; color: #fff; border-radius: 4px; padding: 1px 5px; margin-left: 8px; font-size: 10px; }
        .trade-select { height: 36px; border: 1px solid var(--border-color); border-radius: 6px; background: #1f2937; color: #fff; padding: 0 8px; font-weight: 700; min-width: 78px; }
        .row-action { border: 0; border-radius: 5px; padding: 5px 10px; color: #fff; font-weight: 800; cursor: pointer; font-size: 12px; }
        .row-action.close { background: #ff2f5f; }
        .row-action.edit { background: #f59e0b; padding: 2px 5px; font-size: 10px; }
        .source-badge { background: #10b981; color: #fff; border-radius: 4px; padding: 3px 7px; font-size: 11px; font-weight: 800; }
        .message { min-height: 20px; margin-bottom: 8px; color: var(--text-secondary); font-size: 12px; font-weight: 700; }
        @media (max-width: 1000px) {
            .action-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .trade-select, .action-bar .btn, .price-pill { width: 100%; }
        }

        /* Pagination styles */
        .pagination { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 10px; }
        .pagination .btn-sm { min-width: 32px; text-align: center; }
        .pagination .page-info { font-size: 14px; color: var(--text-secondary); font-weight: 600; }

        /* Date filter section */
        .filter-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }
        .filter-bar input[type='date'] { padding: 6px 10px; border-radius: 6px; border: 1.5px solid var(--border-color); background: var(--bg-primary); color: var(--text-primary); font-size: 12px; }
        .filter-bar select { padding: 6px 10px; border-radius: 6px; border: 1.5px solid var(--border-color); background: var(--bg-primary); color: var(--text-primary); font-size: 12px; }
        #last-update { color: var(--text-secondary); font-size: 11px; text-align: center; margin-top: 10px; }

        /* Theme toggle button — positioned in header */
        .theme-toggle { position: absolute; right: 20px; top: 50%; transform: translateY(-50%); background: var(--btn-bg); color: #fff; border: 1px solid var(--border-color); padding: 8px 14px; border-radius: 20px; cursor: pointer; font-size: 18px; transition: background 0.3s; display: flex; align-items: center; gap: 6px; }
        .theme-toggle:hover { background: var(--btn-hover); }
        .theme-toggle .label { font-size: 11px; }
        .session-actions { position: absolute; left: 20px; top: 50%; transform: translateY(-50%); display: flex; align-items: center; gap: 8px; }
        .logout-link { display: inline-flex; align-items: center; height: 34px; padding: 0 12px; border-radius: 8px; background: #ef4444; color: #fff; font-weight: 800; font-size: 12px; text-decoration: none; }
        .logout-link:hover { background: #dc2626; }

        /* Chart section styles */
        .chart-section { margin-bottom: 15px; }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .chart-tabs { display: flex; gap: 8px; }
        .chart-tab { padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: bold; }
        .chart-tab.btc { background: var(--btn-bg); color: #f7931a; border: 1px solid #f7931a33; }
        .chart-tab.btc.active { background: #f7931a22; border-color: #f7931a; }
        .chart-tab.gold { background: var(--btn-bg); color: #ffd700; border: 1px solid #ffd70033; }
        .chart-tab.gold.active { background: #ffd70022; border-color: #ffd700; }
        .chart-legend { display: flex; gap: 15px; font-size: 11px; color: var(--text-secondary); }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .legend-dot { width: 8px; height: 3px; border-radius: 1px; }
        #chart-container { background: var(--bg-chart); border-radius: 10px; border: 1px solid var(--border-color); overflow: hidden; transition: background 0.3s; }
        #rsi-container { background: var(--bg-chart); border-radius: 0 0 10px 10px; border: 1px solid var(--border-color); border-top: none; overflow: hidden; transition: background 0.3s; }
        .chart-info { display: flex; gap: 20px; padding: 8px 15px; background: var(--bg-chart); border: 1px solid var(--border-color); border-bottom: none; border-radius: 10px 10px 0 0; font-size: 12px; color: var(--text-secondary); transition: background 0.3s; }
        .chart-info .price { font-size: 18px; font-weight: bold; color: var(--text-primary); }
        .chart-info .change { font-size: 13px; }
        .tv-terminal { background: #fff; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        body.dark .tv-terminal { background: var(--bg-card); box-shadow: none; }
        .tv-topbar { display: flex; justify-content: space-between; gap: 10px; align-items: center; padding: 8px 10px; background: #eef6fc; border-bottom: 1px solid #cbd5e1; }
        body.dark .tv-topbar { background: #1e293b; border-color: #334155; }
        .tv-symbol-tabs, .tv-actions, .tv-toolbar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
        .tv-symbol-tab { border: 1px solid #f59e0b; color: #f97316; background: #fff7ed; border-radius: 4px; padding: 6px 13px; font-size: 12px; font-weight: 800; cursor: pointer; }
        .tv-symbol-tab.gold { border-color: #facc15; color: #0369a1; background: #eff6ff; }
        .tv-symbol-tab.active { background: #0ea5e9; color: #fff; border-color: #0284c7; }
        .tv-action-btn { border: 0; border-radius: 5px; color: #fff; padding: 7px 10px; font-size: 12px; font-weight: 800; cursor: pointer; }
        .tv-action-btn.buy { background: #14b8a6; }
        .tv-action-btn.sell { background: #ef4444; }
        .tv-action-btn.dark { background: #1f2937; }
        .tv-action-btn.blue { background: #0b7ce6; }
        .tv-select { height: 31px; border: 0; border-radius: 5px; background: #1f2937; color: #fff; padding: 0 10px; font-size: 12px; font-weight: 800; }
        .tv-toolbar { padding: 7px 10px; background: #fff; border-bottom: 1px solid #e2e8f0; color: #111827; }
        body.dark .tv-toolbar { background: #111827; border-color: #334155; color: #d1d5db; }
        .tv-search { display: flex; align-items: center; gap: 6px; min-width: 110px; font-weight: 800; font-size: 13px; }
        .tv-timeframe { border: 0; background: transparent; color: inherit; padding: 4px 5px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        .tv-timeframe.active { background: #e5e7eb; }
        body.dark .tv-timeframe.active { background: #374151; }
        .tv-tool { border: 0; background: transparent; color: inherit; padding: 4px 6px; border-radius: 4px; cursor: pointer; font-weight: 700; }
        .tv-workspace { display: grid; grid-template-columns: 38px minmax(0, 1fr) 190px; min-height: 560px; }
        .tv-left-tools { border-right: 1px solid #e2e8f0; background: #f8fafc; display: flex; flex-direction: column; align-items: center; gap: 9px; padding-top: 10px; }
        body.dark .tv-left-tools { background: #0f172a; border-color: #334155; }
        .tv-left-tools button { width: 26px; height: 26px; border: 0; border-radius: 4px; background: transparent; color: var(--text-primary); cursor: pointer; font-size: 15px; }
        .tv-left-tools button:hover { background: #e5e7eb; }
        body.dark .tv-left-tools button:hover { background: #334155; }
        .tv-chart-pane { min-width: 0; }
        .tv-indicator-strip { position: absolute; top: 8px; left: 12px; z-index: 3; font-size: 11px; line-height: 1.7; color: #111827; background: rgba(255,255,255,0.72); padding: 3px 6px; border-radius: 4px; pointer-events: none; }
        body.dark .tv-indicator-strip { color: #e5e7eb; background: rgba(15,23,42,0.72); }
        .tv-chart-wrap { position: relative; }
        .tv-side-panel { border-left: 1px solid #e2e8f0; background: #fff; padding: 12px; font-size: 12px; }
        body.dark .tv-side-panel { background: #111827; border-color: #334155; }
        .tv-side-panel h3 { font-size: 13px; color: var(--text-primary); margin-bottom: 14px; }
        .tv-side-symbol { font-weight: 800; margin-bottom: 8px; }
        .tv-side-price { font-size: 25px; color: #dc2626; font-weight: 800; margin: 14px 0 4px; }
        .tv-market-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10b981; margin-right: 6px; }
        .tv-rsi-pane { border-top: 1px solid #e2e8f0; }
        body.dark .tv-rsi-pane { border-color: #334155; }
        .tv-maximized { position: fixed; inset: 8px; z-index: 9999; border-radius: 8px; }
        .tv-maximized .tv-workspace { min-height: calc(100vh - 93px); }
        .tv-maximized #chart-container { height: calc(100vh - 250px) !important; }
        @media (max-width: 900px) {
            .tv-workspace { grid-template-columns: 32px minmax(0, 1fr); }
            .tv-side-panel { display: none; }
            .tv-topbar { align-items: flex-start; flex-direction: column; }
        }

        /* Card shadow for light theme */
        .card { box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        body.dark .card { box-shadow: none; }
    </style>
</head>
<body>
    <div class="header">
        {% if auth_enabled %}
        <div class="session-actions">
            <a class="logout-link" href="/logout">Logout</a>
        </div>
        {% endif %}
        <div class="header-content">
            <h1>Suvshrabani AI Trading System <span class="mode-badge {{ 'live' if mode == 'live' else 'demo' }}">{{ 'LIVE' if mode == 'live' else 'DEMO' }}</span></h1>
            <div class="subtitle">BTC/USDT & XAU/USD | AI-Powered Real-Time Trading {{ '| Delta Exchange' if mode == 'live' else '| Paper Trading' }}</div>
        </div>
        <!-- Light/Dark theme toggle button — saves preference to localStorage -->
        <button class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" title="Switch theme">
            <span id="theme-icon">&#9728;</span><span class="label" id="theme-label">Light</span>
        </button>
    </div>
    <div class="container">
        <!-- Data cards at top — Portfolio, Risk, System Status -->
        <div class="action-bar">
            <button class="btn" onclick="refreshData()">Refresh</button>
            <div class="price-pill btc" id="btc-price-pill">BTC: -- <span class="price-change">--</span></div>
            <div class="price-pill gold" id="gold-price-pill">GOLD: -- <span class="price-change">--</span></div>
            <button class="btn btn-danger" onclick="stopAll()">Stop All</button>
            <button class="btn btn-danger" onclick="stopSymbol('BTC/USDT')">Stop BTC</button>
            <button class="btn btn-danger" onclick="stopSymbol('XAU/USD')">Stop Gold</button>
            <select class="trade-select" id="manual-symbol"><option value="BTC/USDT">BTC</option><option value="XAU/USD">Gold</option></select>
            <select class="trade-select" id="manual-side"><option value="buy">Buy</option><option value="sell">Sell</option></select>
            <select class="trade-select" id="manual-order-type"><option value="market">Market</option></select>
            <select class="trade-select" id="manual-leverage"><option value="1">Lev 1x</option><option value="10">10x</option><option value="25">25x</option><option value="50">50x</option></select>
            <select class="trade-select" id="manual-allocation"><option value="10">10%</option><option value="25">25%</option><option value="50" selected>50%</option><option value="100">100%</option></select>
            <button class="btn btn-success" onclick="executeManualOrder()">Execute</button>
        </div>
        <div class="message" id="action-message"></div>
        <div class="grid" id="metrics-grid">
            <div class="card">
                <h2>Portfolio</h2>
                <div id="portfolio-metrics">Loading...</div>
            </div>
            <div class="card">
                <h2>Risk Exposure</h2>
                <div id="risk-metrics">Loading...</div>
            </div>
            <div class="card">
                <h2>System Status</h2>
                <div id="system-metrics">Loading...</div>
            </div>
        </div>
        <div class="card">
            <h2>Open Positions</h2>
            <table>
                <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry Price</th><th>Entry Time</th><th>Current</th><th>PnL</th><th>PnL %</th><th>Stop Loss</th><th>Source</th><th>Action</th></tr></thead>
                <tbody id="positions-table"><tr><td colspan="11">No open positions</td></tr></tbody>
            </table>
        </div>
        <div class="card" style="margin-top: 15px;">
            <h2>Trade History</h2>
            <!-- Date filter controls for custom date range filtering -->
            <div class="filter-bar">
                <select id="date-preset" onchange="applyDatePreset()">
                    <option value="today">Today</option>
                    <option value="24h" selected>Last 24 Hours</option>
                    <option value="7d">Last 7 Days</option>
                    <option value="30d">Last 30 Days</option>
                    <option value="custom">Custom Range</option>
                </select>
                <input type="date" id="date-from" style="display:none" onchange="filterTrades()">
                <input type="date" id="date-to" style="display:none" onchange="filterTrades()">
                <button class="btn btn-sm btn-outline" onclick="filterTrades()">Apply Filter</button>
                <span class="page-info" id="trade-summary"></span>
            </div>
            <table>
                <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Entry Price</th><th>Exit Price</th><th>PnL</th><th>Reason</th></tr></thead>
                <tbody id="trades-table"><tr><td colspan="7">No trades yet</td></tr></tbody>
            </table>
            <!-- Pagination controls for trade history pages -->
            <div class="pagination" id="trade-pagination"></div>
        </div>

        <!-- TradingView Chart Section — moved to bottom for better workflow -->
        <div class="chart-section tv-terminal" id="tv-terminal" style="margin-top: 15px;">
            <div class="tv-topbar">
                <div class="tv-symbol-tabs">
                    <button class="tv-symbol-tab btc active" onclick="switchChart('BTC/USDT')">BTC/USDT</button>
                    <button class="tv-symbol-tab gold" onclick="switchChart('XAU/USD')">XAU/USD</button>
                </div>
                <div class="tv-actions">
                    <select class="tv-select" id="tv-symbol-select" onchange="switchChart(this.value)">
                        <option value="BTC/USDT">BTC</option>
                        <option value="XAU/USD">Gold</option>
                    </select>
                    <button class="tv-action-btn buy" onclick="executeChartOrder('buy')">TV Buy</button>
                    <button class="tv-action-btn sell" onclick="executeChartOrder('sell')">TV Sell</button>
                    <button class="tv-action-btn dark" onclick="clearChartMarkers()">Clear TV</button>
                    <button class="tv-action-btn blue" onclick="toggleChartMaximize()">Maximize</button>
                </div>
            </div>
            <div class="tv-toolbar">
                <div class="tv-search">Search <span id="chart-symbol">BTC/USDT</span></div>
                <button class="tv-tool" onclick="refreshChart()">+</button>
                <button class="tv-timeframe" onclick="setTimeframe(this)">1m</button>
                <button class="tv-timeframe" onclick="setTimeframe(this)">30m</button>
                <button class="tv-timeframe" onclick="setTimeframe(this)">1h</button>
                <button class="tv-timeframe active" onclick="setTimeframe(this)">5m</button>
                <button class="tv-tool" onclick="refreshChart()">Refresh</button>
                <button class="tv-tool" onclick="toggleIndicators()">Indicators</button>
                <span id="chart-change" class="change">--</span>
            </div>
            <div class="tv-workspace">
                <div class="tv-left-tools">
                    <button title="Crosshair">+</button>
                    <button title="Trend line">/</button>
                    <button title="Horizontal line">--</button>
                    <button title="Brush">~</button>
                    <button title="Measure">[]</button>
                    <button title="Settings">..</button>
                </div>
                <div class="tv-chart-pane">
                    <div class="tv-chart-wrap">
                        <div class="tv-indicator-strip" id="tv-indicators">
                            <div><strong id="tv-title">Bitcoin / TetherUS - 5 - Paper</strong></div>
                            <div>EMA 9 close <span id="tv-ema-fast" class="positive">--</span></div>
                            <div>BB 20 close 2 <span id="tv-bb-values">--</span></div>
                            <div>Vol - BTC <span id="tv-volume">--</span></div>
                        </div>
                        <div id="chart-container" style="height: 430px;"></div>
                    </div>
                    <div class="tv-rsi-pane" id="rsi-container" style="height: 120px;"></div>
                </div>
                <div class="tv-side-panel">
                    <h3 id="tv-panel-symbol">BTCUSDT</h3>
                    <div class="tv-side-symbol" id="tv-panel-name">Bitcoin / TetherUS</div>
                    <div>BINANCE</div>
                    <div>Spot - Crypto</div>
                    <div class="tv-side-price" id="chart-price">--</div>
                    <div id="tv-panel-change" class="negative">--</div>
                    <div style="margin-top: 12px;"><span class="tv-market-dot"></span>Market open</div>
                    <div style="margin-top: 14px; color: var(--text-secondary);">Paper chart powered by app candles</div>
                </div>
            </div>
        </div>

        <div id="last-update">Last updated: --</div>
    </div>
    <script>
        // ========== THEME TOGGLE ==========
        // Light blue (default) / Dark theme switching with localStorage persistence
        let isDark = false;  // Light blue is the default theme

        function loadTheme() {
            // Load saved theme preference from localStorage
            const saved = localStorage.getItem('trading-theme');
            if (saved === 'dark') {
                isDark = true;
                document.body.classList.add('dark');
                document.getElementById('theme-icon').innerHTML = '&#9790;';
                document.getElementById('theme-label').textContent = 'Dark';
            }
        }

        function toggleTheme() {
            isDark = !isDark;
            if (isDark) {
                document.body.classList.add('dark');
                document.getElementById('theme-icon').innerHTML = '&#9790;';
                document.getElementById('theme-label').textContent = 'Dark';
                localStorage.setItem('trading-theme', 'dark');
            } else {
                document.body.classList.remove('dark');
                document.getElementById('theme-icon').innerHTML = '&#9728;';
                document.getElementById('theme-label').textContent = 'Light';
                localStorage.setItem('trading-theme', 'light');
            }
            // Update TradingView chart colors to match the new theme
            applyChartTheme();
        }

        function getChartColors() {
            // Return chart color scheme based on current theme
            if (isDark) {
                return { bg: '#131722', text: '#d1d4dc', grid: '#1e222d', border: '#2B2B43' };
            } else {
                return { bg: '#ffffff', text: '#1e293b', grid: '#e2e8f0', border: '#cbd5e1' };
            }
        }

        function applyChartTheme() {
            // Apply dark or light colors to the TradingView charts
            const c = getChartColors();
            if (mainChart) {
                mainChart.applyOptions({
                    layout: { background: { color: c.bg }, textColor: c.text },
                    grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
                    rightPriceScale: { borderColor: c.border },
                    timeScale: { borderColor: c.border },
                });
            }
            if (rsiChart) {
                rsiChart.applyOptions({
                    layout: { background: { color: c.bg }, textColor: c.text },
                    grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
                    rightPriceScale: { borderColor: c.border },
                });
            }
        }

        // Load theme on page load (before DOMContentLoaded to avoid flash)
        loadTheme();

        // ========== CURRENCY — always display in USD ($), with INR conversion shown separately ==========
        const CUR_SYM = '$';
        const LOCALE = 'en-US';

        // ========== CHART GLOBALS ==========
        let currentSymbol = 'BTC/USDT';
        let mainChart = null;
        let rsiChart = null;
        let candleSeries = null;
        let bbUpperSeries = null;
        let bbMiddleSeries = null;
        let bbLowerSeries = null;
        let volumeSeries = null;
        let rsiSeries = null;
        let rsiUpperLine = null;
        let rsiLowerLine = null;
        let emaFastSeries = null;
        let emaSlowSeries = null;
        let buyMarkers = [];
        let sellMarkers = [];
        let activePriceLines = [];

        // ========== INIT CHARTS ==========
        function initCharts() {
            const chartEl = document.getElementById('chart-container');
            const rsiEl = document.getElementById('rsi-container');
            const c = getChartColors();

            // Main candlestick chart — theme-aware colors
            mainChart = LightweightCharts.createChart(chartEl, {
                width: chartEl.clientWidth,
                height: chartEl.clientHeight || 430,
                layout: { background: { color: c.bg }, textColor: c.text },
                grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: c.border },
                timeScale: { borderColor: c.border, timeVisible: true, secondsVisible: false },
            });

            // Candlestick series
            candleSeries = mainChart.addCandlestickSeries({
                upColor: '#26a69a', downColor: '#ef5350',
                borderUpColor: '#26a69a', borderDownColor: '#ef5350',
                wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            });

            // Bollinger Bands overlay lines
            bbUpperSeries = mainChart.addLineSeries({ color: '#2962FF', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
            bbMiddleSeries = mainChart.addLineSeries({ color: '#FF6D00', lineWidth: 1, lineStyle: 1, priceLineVisible: false, lastValueVisible: false });
            bbLowerSeries = mainChart.addLineSeries({ color: '#2962FF', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });

            // EMA overlay lines (fast=12, slow=26)
            emaFastSeries = mainChart.addLineSeries({ color: '#00E676', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
            emaSlowSeries = mainChart.addLineSeries({ color: '#FF9800', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });

            // Volume histogram on main chart
            volumeSeries = mainChart.addHistogramSeries({
                color: '#26a69a', priceFormat: { type: 'volume' },
                priceScaleId: 'vol', scaleMargins: { top: 0.85, bottom: 0 },
            });

            // RSI sub-chart — theme-aware colors
            rsiChart = LightweightCharts.createChart(rsiEl, {
                width: rsiEl.clientWidth,
                height: rsiEl.clientHeight || 120,
                layout: { background: { color: c.bg }, textColor: c.text },
                grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
                rightPriceScale: { borderColor: c.border },
                timeScale: { borderColor: c.border, visible: false },
            });

            rsiSeries = rsiChart.addLineSeries({ color: '#E040FB', lineWidth: 2, priceLineVisible: false });
            // RSI overbought/oversold reference lines
            rsiUpperLine = rsiChart.addLineSeries({ color: '#ef535066', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
            rsiLowerLine = rsiChart.addLineSeries({ color: '#26a69a66', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });

            // Sync time scales between main chart and RSI
            mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
                if (range) rsiChart.timeScale().setVisibleLogicalRange(range);
            });
            rsiChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
                if (range) mainChart.timeScale().setVisibleLogicalRange(range);
            });

            // Responsive resize
            window.addEventListener('resize', () => {
                resizeCharts();
            });

            // Load initial chart data
            refreshChart();
        }

        // ========== SWITCH SYMBOL ==========
        function switchChart(symbol) {
            currentSymbol = symbol;
            document.querySelectorAll('.tv-symbol-tab').forEach(t => t.classList.remove('active'));
            if (symbol === 'BTC/USDT') document.querySelector('.tv-symbol-tab.btc').classList.add('active');
            else document.querySelector('.tv-symbol-tab.gold').classList.add('active');
            document.getElementById('tv-symbol-select').value = symbol;
            document.getElementById('chart-symbol').textContent = symbol;
            refreshChart();
        }

        function resizeCharts() {
            const chartEl = document.getElementById('chart-container');
            const rsiEl = document.getElementById('rsi-container');
            if (mainChart) mainChart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight || 430 });
            if (rsiChart) rsiChart.applyOptions({ width: rsiEl.clientWidth, height: rsiEl.clientHeight || 120 });
        }

        function setTimeframe(btn) {
            document.querySelectorAll('.tv-timeframe').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            refreshChart();
        }

        function toggleIndicators() {
            const strip = document.getElementById('tv-indicators');
            strip.style.display = strip.style.display === 'none' ? 'block' : 'none';
        }

        function clearChartMarkers() {
            buyMarkers = [];
            sellMarkers = [];
            if (candleSeries) candleSeries.setMarkers([]);
            showActionMessage('Chart markers cleared locally.');
        }

        function toggleChartMaximize() {
            document.getElementById('tv-terminal').classList.toggle('tv-maximized');
            setTimeout(resizeCharts, 50);
        }

        function executeChartOrder(side) {
            const leverageEl = document.getElementById('manual-leverage');
            const allocationEl = document.getElementById('manual-allocation');
            const payload = {
                symbol: currentSymbol,
                side,
                order_type: 'market',
                leverage: Number(leverageEl ? leverageEl.value : 1),
                allocation_pct: Number(allocationEl ? allocationEl.value : 10),
            };
            postAction('/api/manual-order', payload)
                .then(data => {
                    showActionMessage(`TV ${side.toUpperCase()} executed for ${data.symbol} at ${CUR_SYM}${data.price.toFixed(2)}.`);
                    refreshData();
                    refreshChart();
                })
                .catch(err => showActionMessage(err.message, true));
        }

        // ========== LOAD CHART DATA ==========
        function refreshChart() {
            const sym = currentSymbol.replace('/', '-');
            fetch(`/api/chart/${sym}`)
                .then(r => r.json())
                .then(data => renderChart(data))
                .catch(err => console.error('Chart load failed:', err));
        }

        function renderChart(data) {
            const candles = data.candles || [];
            if (candles.length === 0) return;

            // Prepare OHLCV data for candlestick series
            const ohlc = candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }));
            candleSeries.setData(ohlc);

            // Bollinger Bands overlay
            const bbU = candles.filter(c => c.bb_upper > 0).map(c => ({ time: c.time, value: c.bb_upper }));
            const bbM = candles.filter(c => c.bb_middle > 0).map(c => ({ time: c.time, value: c.bb_middle }));
            const bbL = candles.filter(c => c.bb_lower > 0).map(c => ({ time: c.time, value: c.bb_lower }));
            bbUpperSeries.setData(bbU);
            bbMiddleSeries.setData(bbM);
            bbLowerSeries.setData(bbL);

            // EMA fast (12) and slow (26) moving averages overlay
            const emaF = candles.filter(c => c.ema_fast > 0).map(c => ({ time: c.time, value: c.ema_fast }));
            const emaS = candles.filter(c => c.ema_slow > 0).map(c => ({ time: c.time, value: c.ema_slow }));
            emaFastSeries.setData(emaF);
            emaSlowSeries.setData(emaS);

            // Volume histogram — color green for up candles, red for down
            const vol = candles.map(c => ({
                time: c.time, value: c.volume,
                color: c.close >= c.open ? '#26a69a44' : '#ef535044',
            }));
            volumeSeries.setData(vol);

            // RSI line
            const rsiData = candles.filter(c => c.rsi > 0 && c.rsi < 100).map(c => ({ time: c.time, value: c.rsi }));
            rsiSeries.setData(rsiData);

            // RSI reference lines (overbought=65, oversold=35)
            if (rsiData.length > 1) {
                const first = rsiData[0].time;
                const last = rsiData[rsiData.length - 1].time;
                rsiUpperLine.setData([{ time: first, value: 65 }, { time: last, value: 65 }]);
                rsiLowerLine.setData([{ time: first, value: 35 }, { time: last, value: 35 }]);
            }

            // Trade markers — buy (green arrow up) and sell (red arrow down) on candlestick chart
            const markers = (data.markers || []).map(m => ({
                time: m.time,
                position: m.side === 'buy' ? 'belowBar' : 'aboveBar',
                color: m.side === 'buy' ? '#26a69a' : '#ef5350',
                shape: m.side === 'buy' ? 'arrowUp' : 'arrowDown',
                text: m.side === 'buy' ? `BUY ${CUR_SYM}${m.price.toFixed(0)}` : `SELL ${CUR_SYM}${m.price.toFixed(0)}`,
            }));
            candleSeries.setMarkers(markers);

            // Position lines (entry, stop-loss, take-profit) for open positions
            activePriceLines.forEach(line => candleSeries.removePriceLine(line));
            activePriceLines = [];
            if (data.position) {
                const pos = data.position;
                if (pos.entry_price) {
                    activePriceLines.push(candleSeries.createPriceLine({ price: pos.entry_price, color: '#2196F3', lineWidth: 2, lineStyle: 0, title: 'Entry' }));
                }
                if (pos.stop_loss) {
                    activePriceLines.push(candleSeries.createPriceLine({ price: pos.stop_loss, color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'Stop Loss' }));
                }
                if (pos.take_profit) {
                    activePriceLines.push(candleSeries.createPriceLine({ price: pos.take_profit, color: '#26a69a', lineWidth: 1, lineStyle: 2, title: 'Take Profit' }));
                }
            }

            // Update price info bar
            const last = candles[candles.length - 1];
            const prev = candles.length > 1 ? candles[candles.length - 2] : last;
            const change = last.close - prev.close;
            const changePct = (change / prev.close * 100);
            document.getElementById('chart-price').textContent = `${CUR_SYM}${last.close.toLocaleString(LOCALE, {minimumFractionDigits: 2})}`;
            const changeEl = document.getElementById('chart-change');
            changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePct.toFixed(2)}%)`;
            changeEl.className = `change ${change >= 0 ? 'positive' : 'negative'}`;
            const symbolMeta = currentSymbol === 'BTC/USDT'
                ? { title: 'Bitcoin / TetherUS - 5 - Paper', panel: 'BTCUSDT', name: 'Bitcoin / TetherUS', volume: 'BTC' }
                : { title: 'Gold Spot / U.S. Dollar - 5 - Paper', panel: 'XAUUSD', name: 'Gold Spot / U.S. Dollar', volume: 'XAU' };
            document.getElementById('tv-title').textContent = symbolMeta.title;
            document.getElementById('tv-panel-symbol').textContent = symbolMeta.panel;
            document.getElementById('tv-panel-name').textContent = symbolMeta.name;
            document.getElementById('tv-panel-change').textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} ${change >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
            document.getElementById('tv-panel-change').className = change >= 0 ? 'positive' : 'negative';
            document.getElementById('tv-ema-fast').textContent = last.ema_fast ? last.ema_fast.toFixed(2) : '--';
            document.getElementById('tv-bb-values').textContent = last.bb_upper ? `${last.bb_upper.toFixed(2)} ${last.bb_middle.toFixed(2)} ${last.bb_lower.toFixed(2)}` : '--';
            document.getElementById('tv-volume').textContent = `${symbolMeta.volume} ${Number(last.volume || 0).toLocaleString(LOCALE, {maximumFractionDigits: 2})}`;

            // Auto-fit chart to show all data
            mainChart.timeScale().fitContent();
            rsiChart.timeScale().fitContent();
        }

        // ========== DASHBOARD DATA ==========
        function refreshData() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => updateDashboard(data))
                .catch(err => console.error('Refresh failed:', err));
        }

        function routeSymbol(symbol) {
            return symbol.replace('/', '-');
        }

        function showActionMessage(message, isError = false) {
            const el = document.getElementById('action-message');
            el.textContent = message;
            el.className = `message ${isError ? 'negative' : 'positive'}`;
        }

        function postAction(url, body = null) {
            return fetch(url, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: body ? JSON.stringify(body) : null,
            }).then(async r => {
                const data = await r.json();
                if (!r.ok) throw new Error(data.error || 'Action failed');
                return data;
            });
        }

        function stopAll() {
            postAction('/api/halt')
                .then(() => { showActionMessage('All automated trading halted.'); refreshData(); })
                .catch(err => showActionMessage(err.message, true));
        }

        function stopSymbol(symbol) {
            postAction(`/api/halt-symbol/${routeSymbol(symbol)}`)
                .then(() => { showActionMessage(`${symbol} automated entries halted.`); refreshData(); })
                .catch(err => showActionMessage(err.message, true));
        }

        function executeManualOrder() {
            const payload = {
                symbol: document.getElementById('manual-symbol').value,
                side: document.getElementById('manual-side').value,
                order_type: document.getElementById('manual-order-type').value,
                leverage: Number(document.getElementById('manual-leverage').value),
                allocation_pct: Number(document.getElementById('manual-allocation').value),
            };
            postAction('/api/manual-order', payload)
                .then(data => {
                    showActionMessage(`Manual ${payload.side.toUpperCase()} executed for ${data.symbol} at ${CUR_SYM}${data.price.toFixed(2)}.`);
                    refreshData();
                    refreshChart();
                })
                .catch(err => showActionMessage(err.message, true));
        }

        function closePosition(symbol) {
            if (!confirm(`Close ${symbol} position now?`)) return;
            postAction(`/api/close-position/${routeSymbol(symbol)}`)
                .then(() => { showActionMessage(`${symbol} position closed.`); refreshData(); refreshChart(); })
                .catch(err => showActionMessage(err.message, true));
        }

        function editStop(symbol, currentStop) {
            const value = prompt(`New stop loss for ${symbol}`, currentStop || '');
            if (value === null) return;
            const stopLoss = Number(value);
            if (!Number.isFinite(stopLoss) || stopLoss <= 0) {
                showActionMessage('Please enter a valid stop-loss price.', true);
                return;
            }
            postAction(`/api/update-stop/${routeSymbol(symbol)}`, {stop_loss: stopLoss})
                .then(() => { showActionMessage(`${symbol} stop loss updated.`); refreshData(); refreshChart(); })
                .catch(err => showActionMessage(err.message, true));
        }

        function formatPrice(value) {
            return Number(value || 0).toLocaleString(LOCALE, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }

        function updateDashboard(data) {
            const portfolio = data.portfolio || {};
            const inr = portfolio.inr || {};
            const rate = data.currency?.inr_usd_rate || 0;
            const prices = data.prices || {};
            document.getElementById('btc-price-pill').innerHTML = `BTC: ${CUR_SYM}${formatPrice(prices['BTC/USDT'])} <span class="price-change">feed</span>`;
            document.getElementById('gold-price-pill').innerHTML = `GOLD: ${CUR_SYM}${formatPrice(prices['XAU/USD'])} <span class="price-change">feed</span>`;
            const inrValueRow = rate > 0 ? `<div class="metric"><span class="label">Value (INR)</span><span class="value" style="color:#ffd700">&#8377;${(inr.total_value || portfolio.total_value * rate).toLocaleString('en-IN', {minimumFractionDigits: 2})}</span></div>` : '';
            document.getElementById('portfolio-metrics').innerHTML = `
                <div class="metric"><span class="label">Total Value</span><span class="value">${CUR_SYM}${(portfolio.total_value || 0).toLocaleString(LOCALE, {minimumFractionDigits: 2})}</span></div>
                ${inrValueRow}
                <div class="metric"><span class="label">Cash</span><span class="value">${CUR_SYM}${(portfolio.cash_balance || 0).toLocaleString(LOCALE, {minimumFractionDigits: 2})}</span></div>
                <div class="metric"><span class="label">Total Return</span><span class="value ${(portfolio.total_return_pct || 0) >= 0 ? 'positive' : 'negative'}">${(portfolio.total_return_pct || 0).toFixed(2)}%</span></div>
                <div class="metric"><span class="label">Unrealized PnL</span><span class="value ${(portfolio.total_unrealized_pnl || 0) >= 0 ? 'positive' : 'negative'}">${CUR_SYM}${(portfolio.total_unrealized_pnl || 0).toFixed(2)}</span></div>
                <div class="metric"><span class="label">Total Trades</span><span class="value">${portfolio.total_trades || 0}</span></div>
            `;

            const risk = data.risk || {};
            const marginUsedPct = risk.exposure_used_pct || 0;
            const maxExposurePct = risk.max_exposure_pct || 0;
            const leverage = (data.system || {}).leverage || 1;
            const usedNotional = risk.total_notional_exposure_usd || risk.total_exposure_usd || 0;
            const maxNotional = (risk.max_exposure_usd || 0) * leverage;
            const notionalUsedPct = maxNotional > 0 ? (usedNotional / maxNotional * 100) : 0;
            document.getElementById('risk-metrics').innerHTML = `
                <div class="metric"><span class="label">Drawdown</span><span class="value ${(risk.drawdown_pct || 0) < -5 ? 'negative' : 'neutral'}">${(risk.drawdown_pct || 0).toFixed(2)}%</span></div>
                <div class="metric"><span class="label">Max Drawdown</span><span class="value">${(risk.max_drawdown_pct || 0).toFixed(0)}%</span></div>
                <div class="metric"><span class="label">Daily PnL</span><span class="value ${(risk.daily_pnl || 0) >= 0 ? 'positive' : 'negative'}">${CUR_SYM}${(risk.daily_pnl || 0).toFixed(2)}</span></div>
                <div class="metric"><span class="label">Max Daily Loss</span><span class="value">${CUR_SYM}${(risk.max_daily_loss_usd || 0).toFixed(2)} (${(risk.max_daily_loss_pct || 0).toFixed(0)}%)</span></div>
                <div class="metric"><span class="label">Open Positions</span><span class="value">${risk.open_positions || 0}</span></div>
                <div class="metric"><span class="label">Used Exposure</span><span class="value">${CUR_SYM}${usedNotional.toFixed(2)} (${notionalUsedPct.toFixed(1)}%)</span></div>
                <div class="metric"><span class="label">Max Exposure</span><span class="value">${CUR_SYM}${maxNotional.toFixed(2)} (${maxExposurePct.toFixed(0)}% x ${leverage}x)</span></div>
                <div class="metric"><span class="label">Margin Used</span><span class="value">${CUR_SYM}${(risk.total_exposure_usd || 0).toFixed(2)} (${marginUsedPct.toFixed(1)}%)</span></div>
                <div class="metric"><span class="label">Consec. Losses</span><span class="value">${risk.consecutive_losses || 0}</span></div>
            `;

            const system = data.system || {};
            const leverageSelect = document.getElementById('manual-leverage');
            if (leverageSelect && system.leverage) {
                const leverageValue = String(system.leverage);
                if ([...leverageSelect.options].some(option => option.value === leverageValue)) {
                    leverageSelect.value = leverageValue;
                }
            }
            const isActive = !risk.trading_halted;
            const exchangeName = (system.exchange || 'paper').charAt(0).toUpperCase() + (system.exchange || 'paper').slice(1);
            const exchangeColor = {'Binance': '#F0B90B', 'Wazirx': '#2C74F6', 'Delta': '#00D26A', 'Paper': '#888'}[exchangeName] || '#888';
            document.getElementById('system-metrics').innerHTML = `
                <div class="metric"><span class="label">Status</span><span class="value"><span class="status-dot ${isActive ? 'status-active' : 'status-halted'}"></span>${isActive ? 'Active' : 'Halted'}</span></div>
                <div class="metric"><span class="label">Exchange</span><span class="value" style="color:${exchangeColor};font-weight:bold">${exchangeName}</span></div>
                <div class="metric"><span class="label">Mode</span><span class="value">${system.mode || 'paper'}</span></div>
                <div class="metric"><span class="label">Currency</span><span class="value">${system.base_currency || 'USD'}</span></div>
                <div class="metric"><span class="label">Leverage</span><span class="value" style="color:${(system.leverage || 1) > 1 ? '#e74c3c' : '#2ecc71'}">${system.leverage || 1}x</span></div>
                <div class="metric"><span class="label">Uptime</span><span class="value">${system.uptime || '--'}</span></div>
                <div class="metric"><span class="label">Strategy</span><span class="value">${system.active_strategy || '--'}</span></div>
            `;

            const positions = data.portfolio?.open_positions || {};
            const posRows = Object.entries(positions).map(([sym, p]) => {
                const entryTime = p.entry_time ? new Date(p.entry_time).toLocaleString() : '--';
                const stopText = p.stop_loss ? `${CUR_SYM}${p.stop_loss.toFixed(2)}` : '--';
                return `<tr>
                    <td>${sym}</td>
                    <td>${p.side}</td>
                    <td>${p.qty?.toFixed(6) || 0}</td>
                    <td>${CUR_SYM}${p.entry?.toFixed(2) || 0}</td>
                    <td>${entryTime}</td>
                    <td>${CUR_SYM}${p.current?.toFixed(2) || 0}</td>
                    <td class="${p.pnl >= 0 ? 'positive' : 'negative'}">${CUR_SYM}${p.pnl?.toFixed(2) || 0}</td>
                    <td class="${p.pnl_pct >= 0 ? 'positive' : 'negative'}">${p.pnl_pct?.toFixed(2) || 0}%</td>
                    <td>${stopText} <button class="row-action edit" onclick="editStop('${sym}', '${p.stop_loss || ''}')">Edit</button></td>
                    <td><span class="source-badge">${p.source || 'Auto'}</span></td>
                    <td><button class="row-action close" onclick="closePosition('${sym}')">Close</button></td>
                </tr>`;
            }).join('') || '<tr><td colspan="11">No open positions</td></tr>';
            document.getElementById('positions-table').innerHTML = posRows;

            // Store all trades globally for pagination and date filtering
            allTrades = (data.recent_trades || []).slice().reverse();
            filterTrades();

            document.getElementById('last-update').textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
        }

        // ========== TRADE PAGINATION & DATE FILTERING ==========
        let allTrades = [];       // All trades from the API (stored for filtering/pagination)
        let filteredTrades = [];  // Trades after date filter is applied
        let currentPage = 1;      // Current page number
        const tradesPerPage = 15; // Number of trades per page

        // Apply a date preset (today, 24h, 7d, 30d, custom)
        function applyDatePreset() {
            const preset = document.getElementById('date-preset').value;
            const fromEl = document.getElementById('date-from');
            const toEl = document.getElementById('date-to');
            // Show/hide custom date inputs
            if (preset === 'custom') {
                fromEl.style.display = 'inline-block';
                toEl.style.display = 'inline-block';
                return;
            }
            fromEl.style.display = 'none';
            toEl.style.display = 'none';
            filterTrades();
        }

        // Filter trades by selected date range and render the current page
        function filterTrades() {
            const preset = document.getElementById('date-preset').value;
            const now = new Date();
            let fromDate = null;
            let toDate = new Date(now.getTime() + 86400000); // end of today

            if (preset === 'today') {
                fromDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            } else if (preset === '24h') {
                fromDate = new Date(now.getTime() - 86400000);
            } else if (preset === '7d') {
                fromDate = new Date(now.getTime() - 7 * 86400000);
            } else if (preset === '30d') {
                fromDate = new Date(now.getTime() - 30 * 86400000);
            } else if (preset === 'custom') {
                const f = document.getElementById('date-from').value;
                const t = document.getElementById('date-to').value;
                if (f) fromDate = new Date(f);
                if (t) toDate = new Date(new Date(t).getTime() + 86400000);
            }

            // Apply date filter to all trades
            filteredTrades = allTrades.filter(t => {
                if (!t.closed_at) return true;
                const tradeDate = new Date(t.closed_at);
                if (fromDate && tradeDate < fromDate) return false;
                if (toDate && tradeDate > toDate) return false;
                return true;
            });

            currentPage = 1;
            renderTradePage();
        }

        // Render the current page of filtered trades with Entry/Exit price columns
        function renderTradePage() {
            const totalPages = Math.max(1, Math.ceil(filteredTrades.length / tradesPerPage));
            if (currentPage > totalPages) currentPage = totalPages;
            const start = (currentPage - 1) * tradesPerPage;
            const pageTrades = filteredTrades.slice(start, start + tradesPerPage);

            const tradeRows = pageTrades.map(t => {
                const entryStr = t.entry_price ? `${CUR_SYM}${t.entry_price.toFixed(2)}` : '--';
                const exitStr = t.exit_price ? `${CUR_SYM}${t.exit_price.toFixed(2)}` : '--';
                const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
                const pnlStr = t.pnl !== undefined ? `${CUR_SYM}${t.pnl.toFixed(2)}` : '--';
                const timeStr = t.closed_at ? new Date(t.closed_at).toLocaleString() : '--';
                return `<tr><td>${timeStr}</td><td>${t.symbol}</td><td>${t.side}</td><td>${entryStr}</td><td>${exitStr}</td><td class="${pnlClass}">${pnlStr}</td><td>${t.reason || ''}</td></tr>`;
            }).join('') || '<tr><td colspan="7">No trades in selected range</td></tr>';
            document.getElementById('trades-table').innerHTML = tradeRows;

            // Update trade summary text
            document.getElementById('trade-summary').textContent = `${filteredTrades.length} trade(s) found`;

            // Render pagination buttons
            let pagHtml = '';
            if (totalPages > 1) {
                pagHtml += `<button class="btn btn-sm btn-outline" onclick="goToPage(1)" ${currentPage===1?'disabled':''}>First</button>`;
                pagHtml += `<button class="btn btn-sm btn-outline" onclick="goToPage(${currentPage-1})" ${currentPage===1?'disabled':''}>Prev</button>`;
                // Show page numbers around current page
                const startP = Math.max(1, currentPage - 2);
                const endP = Math.min(totalPages, currentPage + 2);
                for (let i = startP; i <= endP; i++) {
                    pagHtml += `<button class="btn btn-sm ${i===currentPage?'':'btn-outline'}" onclick="goToPage(${i})">${i}</button>`;
                }
                pagHtml += `<button class="btn btn-sm btn-outline" onclick="goToPage(${currentPage+1})" ${currentPage===totalPages?'disabled':''}>Next</button>`;
                pagHtml += `<button class="btn btn-sm btn-outline" onclick="goToPage(${totalPages})" ${currentPage===totalPages?'disabled':''}>Last</button>`;
                pagHtml += `<span class="page-info">Page ${currentPage} of ${totalPages}</span>`;
            }
            document.getElementById('trade-pagination').innerHTML = pagHtml;
        }

        function goToPage(page) {
            currentPage = page;
            renderTradePage();
        }

        // ========== INIT ==========
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            refreshData();
            // Auto-refresh data and chart every 10 seconds
            setInterval(() => { refreshData(); refreshChart(); }, 10000);
        });
    </script>
</body>
</html>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Suvshrabani AI Trading System - Login</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #eef2f7;
            color: #1e293b;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
        }
        .login-panel {
            width: min(420px, 100%);
            background: #ffffff;
            border: 1px solid #c7d2e0;
            border-radius: 10px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
            padding: 28px;
        }
        h1 { color: #2563eb; font-size: 24px; line-height: 1.2; text-align: center; margin-bottom: 6px; }
        .subtitle { color: #64748b; text-align: center; font-size: 14px; font-weight: 600; margin-bottom: 24px; }
        label { display: block; color: #334155; font-size: 13px; font-weight: 800; margin-bottom: 6px; }
        input {
            width: 100%;
            height: 42px;
            border: 1px solid #c7d2e0;
            border-radius: 8px;
            padding: 0 12px;
            margin-bottom: 14px;
            font-size: 15px;
            outline: none;
        }
        input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12); }
        button {
            width: 100%;
            height: 42px;
            border: 0;
            border-radius: 8px;
            background: #2563eb;
            color: #fff;
            font-size: 15px;
            font-weight: 800;
            cursor: pointer;
        }
        button:hover { background: #1d4ed8; }
        .error {
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fecaca;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 14px;
            text-align: center;
        }
    </style>
</head>
<body>
    <form class="login-panel" method="post" action="/login">
        <h1>Suvshrabani AI Trading System</h1>
        <div class="subtitle">Secure dashboard access</div>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        <label for="username">Username</label>
        <input id="username" name="username" type="text" autocomplete="username" autofocus required>
        <label for="password">Password</label>
        <input id="password" name="password" type="password" autocomplete="current-password" required>
        <button type="submit">Login</button>
    </form>
</body>
</html>
"""


class TradingDashboard:
    """
    Flask-based monitoring dashboard and REST API.

    Provides:
        - Web dashboard at / for visual monitoring
        - REST API at /api/* for external control
        - WebSocket updates via SocketIO for real-time data push
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
    ):
        """
        Initialize the dashboard.

        Args:
            host: Host to bind the Flask server
            port: Port for the Flask server
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get("DASHBOARD_SECRET_KEY") or secrets.token_hex(32)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self._dashboard_username = os.environ.get("DASHBOARD_USERNAME", "suvendu")
        self._dashboard_password = os.environ.get("DASHBOARD_PASSWORD", "")
        self._auth_enabled = os.environ.get("DASHBOARD_AUTH_ENABLED", "").lower() in ("1", "true", "yes")
        if self._dashboard_password and os.environ.get("DASHBOARD_AUTH_ENABLED") is None:
            self._auth_enabled = True
        if self._auth_enabled and not self._dashboard_password:
            logger.warning("DASHBOARD_AUTH_ENABLED is true but DASHBOARD_PASSWORD is empty; dashboard login disabled")
            self._auth_enabled = False

        # References to trading system components (set via set_components)
        self._portfolio = None
        self._risk_manager = None
        self._strategy = None
        self._mode = "paper"
        self._start_time = datetime.now(timezone.utc)

        # Currency converter for dual-currency display (INR + USD)
        self._currency_converter = None
        self._base_currency = "USD"

        # Active exchange name so the user can see which broker executes trades
        self._exchange_name = "paper"

        # Chart data storage — stores OHLCV candles + indicators per symbol for charting
        self._candle_data = {}
        # Trade markers for chart overlay (buy/sell points)
        self._trade_markers = []

        self._setup_routes()

    def set_components(self, portfolio=None, risk_manager=None, strategy=None, mode: str = "paper",
                       currency_converter=None, base_currency: str = "USD", exchange_name: str = "paper",
                       leverage: int = 1):
        """
        Inject trading system component references for API access.

        Args:
            portfolio: Portfolio instance
            risk_manager: RiskManager instance
            strategy: Active strategy instance
            mode: Trading mode string
            currency_converter: CurrencyConverter instance for INR/USD conversion
            base_currency: User's selected base currency ("USD" or "INR")
            exchange_name: Active exchange broker name ("binance", "wazirx", "delta", "paper")
            leverage: Leverage multiplier (1 = no leverage, 25 = 25x)
        """
        self._portfolio = portfolio
        self._risk_manager = risk_manager
        self._strategy = strategy
        self._mode = mode
        self._currency_converter = currency_converter
        self._base_currency = base_currency
        self._exchange_name = exchange_name
        self._leverage = leverage

    def _latest_price(self, symbol: str) -> Optional[float]:
        """Return the latest dashboard candle price, falling back to portfolio mark price."""
        candles = self._candle_data.get(symbol, [])
        if candles:
            return float(candles[-1].get("close", 0) or 0)
        if self._portfolio and symbol in self._portfolio.positions:
            return float(self._portfolio.positions[symbol].current_price or 0)
        return None

    def _setup_routes(self):
        """Register all Flask routes for dashboard and API."""

        @self.app.before_request
        def require_dashboard_login():
            """Route browser/API traffic through the app login when enabled."""
            if not self._auth_enabled:
                return None
            allowed_endpoints = {"login", "logout", "api_health", "static"}
            if request.endpoint in allowed_endpoints:
                return None
            if session.get("dashboard_authenticated"):
                return None
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login", next=request.full_path.rstrip("?")))

        @self.app.route("/login", methods=["GET", "POST"])
        def login():
            """Render and process the dashboard login form."""
            if not self._auth_enabled:
                return redirect(url_for("index"))
            error = None
            if request.method == "POST":
                username = request.form.get("username", "")
                password = request.form.get("password", "")
                username_ok = secrets.compare_digest(username, self._dashboard_username)
                password_ok = secrets.compare_digest(password, self._dashboard_password)
                if username_ok and password_ok:
                    session["dashboard_authenticated"] = True
                    return redirect(request.args.get("next") or url_for("index"))
                error = "Invalid username or password"
            return render_template_string(LOGIN_TEMPLATE, error=error)

        @self.app.route("/logout")
        def logout():
            """Clear the dashboard session and return to login."""
            session.clear()
            return redirect(url_for("login"))

        @self.app.route("/")
        def index():
            """Serve the monitoring dashboard HTML page with mode-aware theme."""
            # Pass trading mode and base currency to template so theme and currency symbol change
            return render_template_string(
                DASHBOARD_TEMPLATE,
                mode=self._mode,
                base_currency=self._base_currency,
                auth_enabled=self._auth_enabled,
            )

        @self.app.route("/api/status")
        def api_status():
            """Return full system status with dual-currency values (USD + INR)."""
            # Get INR/USD rate for dual-currency display
            inr_rate = 0.0
            if self._currency_converter:
                inr_rate = self._currency_converter.get_rate("USD", "INR")

            portfolio_data = self._portfolio.get_summary(inr_rate=inr_rate) if self._portfolio else {}
            risk_data = self._risk_manager.get_risk_summary() if self._risk_manager else {}
            # Return all trade history for full pagination and date filtering in dashboard
            recent_trades = self._portfolio.trade_history if self._portfolio else []

            uptime = str(datetime.now(timezone.utc) - self._start_time).split(".")[0]
            prices = {
                symbol: candles[-1]["close"]
                for symbol, candles in self._candle_data.items()
                if candles
            }

            return jsonify({
                "portfolio": portfolio_data,
                "risk": risk_data,
                "recent_trades": recent_trades,
                "prices": prices,
                "system": {
                    "mode": self._mode,
                    "uptime": uptime,
                    "active_strategy": self._strategy.get_name() if self._strategy else "none",
                    "base_currency": self._base_currency,
                    "exchange": self._exchange_name,
                    "leverage": getattr(self, '_leverage', 1),
                },
                "currency": {
                    "base": self._base_currency,
                    "inr_usd_rate": inr_rate,
                },
            })

        @self.app.route("/api/portfolio")
        def api_portfolio():
            """Return portfolio details."""
            if not self._portfolio:
                return jsonify({"error": "Portfolio not initialized"}), 500
            return jsonify(self._portfolio.get_summary())

        @self.app.route("/api/risk")
        def api_risk():
            """Return risk metrics."""
            if not self._risk_manager:
                return jsonify({"error": "Risk manager not initialized"}), 500
            return jsonify(self._risk_manager.get_risk_summary())

        @self.app.route("/api/trades")
        def api_trades():
            """Return trade history."""
            if not self._portfolio:
                return jsonify({"error": "Portfolio not initialized"}), 500
            limit = request.args.get("limit", 50, type=int)
            return jsonify(self._portfolio.trade_history[-limit:])

        @self.app.route("/api/equity")
        def api_equity():
            """Return equity curve data."""
            if not self._portfolio:
                return jsonify({"error": "Portfolio not initialized"}), 500
            return jsonify(self._portfolio.equity_curve[-500:])

        @self.app.route("/api/halt", methods=["POST"])
        def api_halt():
            """Emergency halt trading."""
            if self._risk_manager:
                self._risk_manager.trading_halted = True
                self._risk_manager.halt_reason = "Manual halt via API"
                logger.warning("Trading halted via API")
                return jsonify({"status": "halted"})
            return jsonify({"error": "Risk manager not available"}), 500

        @self.app.route("/api/resume", methods=["POST"])
        def api_resume():
            """Resume trading after a halt."""
            if self._risk_manager:
                self._risk_manager.trading_halted = False
                self._risk_manager.halt_reason = None
                logger.info("Trading resumed via API")
                return jsonify({"status": "resumed"})
            return jsonify({"error": "Risk manager not available"}), 500

        @self.app.route("/api/halt-symbol/<symbol_raw>", methods=["POST"])
        def api_halt_symbol(symbol_raw):
            """Halt new trades for one symbol without stopping the full system."""
            if not self._risk_manager:
                return jsonify({"error": "Risk manager not available"}), 500
            symbol = symbol_raw.replace("-", "/")
            if hasattr(self._risk_manager, "halt_symbol"):
                self._risk_manager.halt_symbol(symbol, "Manual dashboard halt")
            else:
                self._risk_manager.symbol_halts = getattr(self._risk_manager, "symbol_halts", {})
                self._risk_manager.symbol_halts[symbol] = "Manual dashboard halt"
            return jsonify({"status": "halted", "symbol": symbol})

        @self.app.route("/api/resume-symbol/<symbol_raw>", methods=["POST"])
        def api_resume_symbol(symbol_raw):
            """Resume new trades for one symbol."""
            if not self._risk_manager:
                return jsonify({"error": "Risk manager not available"}), 500
            symbol = symbol_raw.replace("-", "/")
            if hasattr(self._risk_manager, "resume_symbol"):
                self._risk_manager.resume_symbol(symbol)
            else:
                getattr(self._risk_manager, "symbol_halts", {}).pop(symbol, None)
            return jsonify({"status": "resumed", "symbol": symbol})

        @self.app.route("/api/manual-order", methods=["POST"])
        def api_manual_order():
            """Open a manual paper position from the dashboard controls."""
            if self._mode != "paper":
                return jsonify({"error": "Manual dashboard orders are enabled only in paper mode"}), 400
            if not self._portfolio or not self._risk_manager:
                return jsonify({"error": "Portfolio or risk manager not available"}), 500

            payload = request.get_json(silent=True) or {}
            symbol = str(payload.get("symbol", "BTC/USDT"))
            side = str(payload.get("side", "buy")).lower()
            leverage = max(1.0, float(payload.get("leverage", getattr(self, "_leverage", 1) or 1)))
            allocation_pct = min(100.0, max(1.0, float(payload.get("allocation_pct", 10))))
            price = self._latest_price(symbol)
            if not price or price <= 0:
                return jsonify({"error": f"No current price available for {symbol} yet"}), 400
            if side not in ("buy", "sell"):
                return jsonify({"error": "Side must be buy or sell"}), 400
            if symbol in self._portfolio.positions:
                return jsonify({"error": f"{symbol} already has an open position. Close it before opening another manual trade."}), 400

            margin_usd = self._portfolio.total_value * allocation_pct / 100
            allowed, reason = self._risk_manager.can_trade(symbol, side, margin_usd)
            if not allowed:
                return jsonify({"error": reason}), 400

            notional_usd = margin_usd * leverage
            quantity = notional_usd / price
            stop_loss = self._risk_manager.calculate_stop_loss(
                price, side, position_notional_usd=notional_usd
            )
            fixed_take_profit = getattr(self._risk_manager.config, "fixed_take_profit_usd", 0)
            dollar_risk_enabled = (
                getattr(self._risk_manager.config, "fixed_stop_loss_usd", 0) > 0
                or getattr(self._risk_manager.config, "trailing_stop_activation_usd", 0) > 0
            )
            if fixed_take_profit > 0:
                take_profit = price + fixed_take_profit if side == "buy" else price - fixed_take_profit
            elif dollar_risk_enabled:
                take_profit = None
            else:
                take_profit = self._risk_manager.calculate_take_profit(price, side, stop_loss)
            position = self._portfolio.open_position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            setattr(position, "source", "Manual")
            self._risk_manager.register_position(
                symbol, side, margin_usd, price, stop_loss, take_profit, notional_usd=notional_usd
            )
            self.add_trade_marker(symbol, "buy" if side == "buy" else "sell", price)
            return jsonify({
                "status": "executed",
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": position.quantity,
                "margin_usd": margin_usd,
                "notional_usd": notional_usd,
                "leverage": leverage,
            })

        @self.app.route("/api/close-position/<symbol_raw>", methods=["POST"])
        def api_close_position(symbol_raw):
            """Close an open paper position from the dashboard table."""
            if not self._portfolio:
                return jsonify({"error": "Portfolio not available"}), 500
            symbol = symbol_raw.replace("-", "/")
            price = self._latest_price(symbol)
            if not price or price <= 0:
                return jsonify({"error": f"No current price available for {symbol}"}), 400
            trade = self._portfolio.close_position(symbol, price, reason="manual_dashboard_close")
            if not trade:
                return jsonify({"error": f"No open position for {symbol}"}), 404
            if self._risk_manager:
                self._risk_manager.close_position(symbol, price)
            self.add_trade_marker(symbol, "sell" if trade.get("side") == "buy" else "buy", price)
            return jsonify({"status": "closed", "symbol": symbol, "trade": trade})

        @self.app.route("/api/update-stop/<symbol_raw>", methods=["POST"])
        def api_update_stop(symbol_raw):
            """Update stop-loss price for an open paper position."""
            if not self._portfolio:
                return jsonify({"error": "Portfolio not available"}), 500
            symbol = symbol_raw.replace("-", "/")
            payload = request.get_json(silent=True) or {}
            stop_loss = float(payload.get("stop_loss", 0) or 0)
            if stop_loss <= 0:
                return jsonify({"error": "Stop loss must be greater than zero"}), 400
            if symbol not in self._portfolio.positions:
                return jsonify({"error": f"No open position for {symbol}"}), 404
            self._portfolio.positions[symbol].stop_loss = stop_loss
            if self._risk_manager and symbol in self._risk_manager.open_positions:
                self._risk_manager.open_positions[symbol]["stop_loss"] = stop_loss
            return jsonify({"status": "updated", "symbol": symbol, "stop_loss": stop_loss})

        @self.app.route("/api/health")
        def api_health():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": self._mode,
                "base_currency": self._base_currency,
                "exchange": self._exchange_name,
            })

        @self.app.route("/api/currency")
        def api_currency():
            """Return current exchange rates and currency configuration."""
            inr_rate = 0.0
            if self._currency_converter:
                inr_rate = self._currency_converter.get_rate("USD", "INR")
            return jsonify({
                "base_currency": self._base_currency,
                "rates": {
                    "USD_INR": inr_rate,
                    "INR_USD": 1.0 / inr_rate if inr_rate > 0 else 0,
                },
                "supported_currencies": ["USD", "INR"],
            })

        @self.app.route("/api/chart/<symbol_raw>")
        def api_chart(symbol_raw):
            """Return OHLCV + indicator data for TradingView chart rendering."""
            # Convert URL-safe symbol (BTC-USDT) back to normal format (BTC/USDT)
            symbol = symbol_raw.replace("-", "/")
            candles = self._candle_data.get(symbol, [])
            # Filter trade markers for this symbol
            markers = [m for m in self._trade_markers if m["symbol"] == symbol]
            # Get open position info for stop-loss/take-profit lines
            position = None
            if self._portfolio and symbol in self._portfolio.positions:
                pos = self._portfolio.positions[symbol]
                position = {
                    "entry_price": pos.entry_price,
                    "stop_loss": getattr(pos, "stop_loss", None),
                    "take_profit": getattr(pos, "take_profit", None),
                    "side": "long",
                    "quantity": pos.quantity,
                }
            return jsonify({
                "candles": candles,
                "markers": markers,
                "position": position,
                "symbol": symbol,
            })

    def update_candle_data(self, symbol: str, df):
        """
        Store latest OHLCV candle data + indicators for chart rendering.
        Called from the main trading loop after each data fetch.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            df: DataFrame with OHLCV + indicator columns (enriched data)
        """
        import numpy as np
        candles = []
        for _, row in df.iterrows():
            ts = int(row.name.timestamp()) if hasattr(row.name, 'timestamp') else 0
            candles.append({
                "time": ts,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
                # Bollinger Bands for overlay
                "bb_upper": float(row.get("bb_upper", 0)) if not np.isnan(row.get("bb_upper", 0)) else 0,
                "bb_middle": float(row.get("bb_middle", 0)) if not np.isnan(row.get("bb_middle", 0)) else 0,
                "bb_lower": float(row.get("bb_lower", 0)) if not np.isnan(row.get("bb_lower", 0)) else 0,
                # RSI for sub-chart
                "rsi": float(row.get("rsi", 50)) if not np.isnan(row.get("rsi", 50)) else 50,
                # EMA lines
                "ema_fast": float(row.get("ema_12", 0)) if not np.isnan(row.get("ema_12", 0)) else 0,
                "ema_slow": float(row.get("ema_26", 0)) if not np.isnan(row.get("ema_26", 0)) else 0,
                # Volume ratio
                "volume_ratio": float(row.get("volume_ratio", 0)) if not np.isnan(row.get("volume_ratio", 0)) else 0,
            })
        self._candle_data[symbol] = candles

    def add_trade_marker(self, symbol: str, side: str, price: float, timestamp=None):
        """
        Add a buy/sell trade marker for chart display.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            price: Execution price
            timestamp: Trade time (defaults to now)
        """
        ts = int(timestamp.timestamp()) if timestamp else int(datetime.now(timezone.utc).timestamp())
        self._trade_markers.append({
            "time": ts,
            "symbol": symbol,
            "side": side,
            "price": price,
        })
        # Keep last 200 markers
        if len(self._trade_markers) > 200:
            self._trade_markers = self._trade_markers[-200:]

    def start(self, threaded: bool = True):
        """
        Start the dashboard server.

        Args:
            threaded: If True, run in a background thread
        """
        if threaded:
            thread = threading.Thread(
                target=self._run_server,
                daemon=True,
            )
            thread.start()
            logger.info(f"Dashboard started at http://{self.host}:{self.port}")
        else:
            self._run_server()

    def _run_server(self):
        """Run the Flask/SocketIO server."""
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )

    def broadcast_update(self, data: dict):
        """Push real-time update to all connected dashboard clients."""
        self.socketio.emit("update", data)
