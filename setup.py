"""
Setup script for the Automated Trading System.
Supports BTC/USDT and XAU/USD trading with full automation.
"""

from setuptools import setup, find_packages

setup(
    name="automated-trading-system",
    version="1.0.0",
    description="Production-grade automated trading system for BTC/USDT and XAU/USD",
    author="Automated Trading Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "ta>=0.11.0",
        "requests>=2.31.0",
        "websocket-client>=1.6.0",
        "aiohttp>=3.9.0",
        "schedule>=1.2.0",
        "APScheduler>=3.10.0",
        "sqlalchemy>=2.0.0",
        "flask>=3.0.0",
        "flask-socketio>=5.3.0",
        "matplotlib>=3.7.0",
        "jinja2>=3.1.0",
        "loguru>=0.7.0",
        "rich>=13.0.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-system=main:main",
        ],
    },
)
