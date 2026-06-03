"""
Real-time monitoring dashboard and REST API.
Provides a Flask-based web dashboard for monitoring trading activity,
PnL, open positions, risk exposure, and system health.
Also exposes a REST API for external control and integration.
Uses TradingView Lightweight Charts (open-source) for interactive candlestick charting.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, jsonify, render_template_string, request
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

        /* Card shadow for light theme */
        .card { box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        body.dark .card { box-shadow: none; }
    </style>
</head>
<body>
    <div class="header">
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
        <button class="btn" onclick="refreshData()" style="margin-bottom:10px">Refresh Data</button>
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
                <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Current</th><th>PnL</th><th>PnL %</th></tr></thead>
                <tbody id="positions-table"><tr><td colspan="7">No open positions</td></tr></tbody>
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
        <div class="chart-section" style="margin-top: 15px;">
            <div class="chart-header">
                <div class="chart-tabs">
                    <div class="chart-tab btc active" onclick="switchChart('BTC/USDT')">BTC/USDT</div>
                    <div class="chart-tab gold" onclick="switchChart('XAU/USD')">XAU/USD</div>
                </div>
                <div class="chart-legend">
                    <div class="legend-item"><div class="legend-dot" style="background:#2962FF"></div>BB Upper</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#FF6D00"></div>BB Middle</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#2962FF"></div>BB Lower</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#00E676"></div>EMA 12</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#FF9800"></div>EMA 26</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#26a69a;height:8px;width:8px;border-radius:50%"></div>Buy</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#ef5350;height:8px;width:8px;border-radius:50%"></div>Sell</div>
                </div>
                <button class="btn" onclick="refreshChart()">Refresh Chart</button>
            </div>
            <div class="chart-info" id="chart-info">
                <div><span id="chart-symbol">BTC/USDT</span> <span class="price" id="chart-price">--</span> <span class="change" id="chart-change">--</span></div>
            </div>
            <div id="chart-container" style="height: 400px;"></div>
            <div id="rsi-container" style="height: 120px;"></div>
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

        // ========== INIT CHARTS ==========
        function initCharts() {
            const chartEl = document.getElementById('chart-container');
            const rsiEl = document.getElementById('rsi-container');
            const c = getChartColors();

            // Main candlestick chart — theme-aware colors
            mainChart = LightweightCharts.createChart(chartEl, {
                width: chartEl.clientWidth,
                height: 400,
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
                height: 120,
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
                mainChart.applyOptions({ width: chartEl.clientWidth });
                rsiChart.applyOptions({ width: rsiEl.clientWidth });
            });

            // Load initial chart data
            refreshChart();
        }

        // ========== SWITCH SYMBOL ==========
        function switchChart(symbol) {
            currentSymbol = symbol;
            document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            if (symbol === 'BTC/USDT') document.querySelector('.chart-tab.btc').classList.add('active');
            else document.querySelector('.chart-tab.gold').classList.add('active');
            document.getElementById('chart-symbol').textContent = symbol;
            refreshChart();
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
            if (data.position) {
                const pos = data.position;
                // Clear previous price lines by setting data again
                if (pos.entry_price) {
                    candleSeries.createPriceLine({ price: pos.entry_price, color: '#2196F3', lineWidth: 2, lineStyle: 0, title: 'Entry' });
                }
                if (pos.stop_loss) {
                    candleSeries.createPriceLine({ price: pos.stop_loss, color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'Stop Loss' });
                }
                if (pos.take_profit) {
                    candleSeries.createPriceLine({ price: pos.take_profit, color: '#26a69a', lineWidth: 1, lineStyle: 2, title: 'Take Profit' });
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

        function updateDashboard(data) {
            const portfolio = data.portfolio || {};
            const inr = portfolio.inr || {};
            const rate = data.currency?.inr_usd_rate || 0;
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
            document.getElementById('risk-metrics').innerHTML = `
                <div class="metric"><span class="label">Drawdown</span><span class="value ${(risk.drawdown_pct || 0) < -5 ? 'negative' : 'neutral'}">${(risk.drawdown_pct || 0).toFixed(2)}%</span></div>
                <div class="metric"><span class="label">Daily PnL</span><span class="value ${(risk.daily_pnl || 0) >= 0 ? 'positive' : 'negative'}">${CUR_SYM}${(risk.daily_pnl || 0).toFixed(2)}</span></div>
                <div class="metric"><span class="label">Open Positions</span><span class="value">${risk.open_positions || 0}</span></div>
                <div class="metric"><span class="label">Exposure</span><span class="value">${CUR_SYM}${(risk.total_exposure_usd || 0).toFixed(2)}</span></div>
                <div class="metric"><span class="label">Consec. Losses</span><span class="value">${risk.consecutive_losses || 0}</span></div>
            `;

            const system = data.system || {};
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
            const posRows = Object.entries(positions).map(([sym, p]) =>
                `<tr><td>${sym}</td><td>${p.side}</td><td>${p.qty?.toFixed(6) || 0}</td><td>${CUR_SYM}${p.entry?.toFixed(2) || 0}</td><td>${CUR_SYM}${p.current?.toFixed(2) || 0}</td><td class="${p.pnl >= 0 ? 'positive' : 'negative'}">${CUR_SYM}${p.pnl?.toFixed(2) || 0}</td><td class="${p.pnl_pct >= 0 ? 'positive' : 'negative'}">${p.pnl_pct?.toFixed(2) || 0}%</td></tr>`
            ).join('') || '<tr><td colspan="7">No open positions</td></tr>';
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
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

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

    def _setup_routes(self):
        """Register all Flask routes for dashboard and API."""

        @self.app.route("/")
        def index():
            """Serve the monitoring dashboard HTML page with mode-aware theme."""
            # Pass trading mode and base currency to template so theme and currency symbol change
            return render_template_string(DASHBOARD_TEMPLATE, mode=self._mode, base_currency=self._base_currency)

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

            return jsonify({
                "portfolio": portfolio_data,
                "risk": risk_data,
                "recent_trades": recent_trades,
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
