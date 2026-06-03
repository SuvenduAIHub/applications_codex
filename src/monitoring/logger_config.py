"""
Logging and error handling configuration.
Sets up structured logging with rotation, trade-specific logging,
and failure recovery mechanisms.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

from loguru import logger


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    rotation_mb: int = 50,
    retention_count: int = 10,
) -> None:
    """
    Configure the application logging system.
    Uses loguru for structured, rotated log output.

    Creates separate log files for:
        - General system logs (trading.log)
        - Trade-specific logs (trades.log)
        - Error logs (errors.log)

    Args:
        log_file: Path to the main log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation_mb: Log rotation size in megabytes
        retention_count: Number of rotated log files to keep
    """
    # Determine log directory
    if log_file:
        log_dir = os.path.dirname(log_file)
    else:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

    os.makedirs(log_dir, exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # Console output with colorized formatting
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Main system log file with rotation
    main_log = os.path.join(log_dir, "trading.log")
    logger.add(
        main_log,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=f"{rotation_mb} MB",
        retention=retention_count,
        compression="zip",
        enqueue=True,  # Thread-safe logging
    )

    # Trade-specific log (only trade-related messages)
    trade_log = os.path.join(log_dir, "trades.log")
    logger.add(
        trade_log,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        rotation=f"{rotation_mb} MB",
        retention=retention_count,
        filter=lambda record: "trade" in record["message"].lower() or "position" in record["message"].lower(),
    )

    # Error-only log for quick issue identification
    error_log = os.path.join(log_dir, "errors.log")
    logger.add(
        error_log,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}\n{exception}",
        rotation=f"{rotation_mb} MB",
        retention=retention_count,
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logging initialized: level={log_level}, dir={log_dir}")


class TradeLogger:
    """
    Specialized logger for trade events.
    Provides structured logging for all trade-related activities.
    """

    @staticmethod
    def log_signal(symbol: str, signal: str, confidence: float, strategy: str, price: float) -> None:
        """Log a trading signal generation event."""
        logger.info(
            f"SIGNAL | {symbol} | {signal} | confidence={confidence:.2f} | "
            f"strategy={strategy} | price={price:.2f}"
        )

    @staticmethod
    def log_order_submitted(order_id: str, symbol: str, side: str, qty: float, price: float, order_type: str) -> None:
        """Log an order submission event."""
        logger.info(
            f"ORDER_SUBMITTED | {order_id} | {symbol} | {side} | "
            f"qty={qty:.6f} | price={price:.2f} | type={order_type}"
        )

    @staticmethod
    def log_order_filled(order_id: str, symbol: str, side: str, qty: float, fill_price: float, commission: float) -> None:
        """Log an order fill event."""
        logger.info(
            f"ORDER_FILLED | {order_id} | {symbol} | {side} | "
            f"qty={qty:.6f} | fill_price={fill_price:.2f} | commission={commission:.2f}"
        )

    @staticmethod
    def log_position_opened(symbol: str, side: str, size: float, entry: float, sl: float, tp: float) -> None:
        """Log a new position opening."""
        logger.info(
            f"POSITION_OPENED | {symbol} | {side} | size=${size:,.2f} | "
            f"entry={entry:.2f} | SL={sl:.2f} | TP={tp:.2f}"
        )

    @staticmethod
    def log_position_closed(symbol: str, pnl: float, pnl_pct: float, reason: str) -> None:
        """Log a position closure."""
        logger.info(
            f"POSITION_CLOSED | {symbol} | PnL=${pnl:,.2f} ({pnl_pct:.2f}%) | reason={reason}"
        )

    @staticmethod
    def log_risk_alert(alert_type: str, message: str) -> None:
        """Log a risk management alert."""
        logger.warning(f"RISK_ALERT | {alert_type} | {message}")

    @staticmethod
    def log_system_error(component: str, error: str) -> None:
        """Log a system error with component identification."""
        logger.error(f"SYSTEM_ERROR | {component} | {error}")


class FailureRecovery:
    """
    Handles system failure recovery.
    Saves and restores system state to allow graceful restart
    after crashes or unexpected shutdowns.
    """

    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize the failure recovery system.

        Args:
            state_dir: Directory to store recovery state files
        """
        self.state_dir = state_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "state"
        )
        os.makedirs(self.state_dir, exist_ok=True)

    def save_state(self, state: dict) -> str:
        """
        Save the current system state to disk for recovery.

        Args:
            state: Dictionary containing system state to persist

        Returns:
            Path to the saved state file
        """
        import json
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.state_dir, f"state_{timestamp}.json")

        # Convert non-serializable types
        serializable = self._make_serializable(state)

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"System state saved to {filepath}")
        return filepath

    def load_latest_state(self) -> Optional[dict]:
        """
        Load the most recent saved state for recovery.

        Returns:
            State dictionary, or None if no state files exist
        """
        import json
        import glob

        state_files = sorted(glob.glob(os.path.join(self.state_dir, "state_*.json")))
        if not state_files:
            logger.info("No recovery state files found")
            return None

        latest = state_files[-1]
        with open(latest, "r") as f:
            state = json.load(f)

        logger.info(f"Loaded recovery state from {latest}")
        return state

    @staticmethod
    def _make_serializable(obj):
        """Recursively convert objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: FailureRecovery._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FailureRecovery._make_serializable(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum
            return obj.value
        return obj
