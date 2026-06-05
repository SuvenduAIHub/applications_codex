"""
Local data storage and caching layer for market data.
Uses SQLite for persistent storage and provides methods to
store, retrieve, and manage historical OHLCV data.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for ORM models."""
    pass


class OHLCVRecord(Base):
    """
    Database model for storing OHLCV candlestick data.
    Each record represents one candle for a specific symbol and timeframe.
    """
    __tablename__ = "ohlcv_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)        # e.g., "BTC/USDT"
    timeframe = Column(String(10), nullable=False, index=True)     # e.g., "1h"
    timestamp = Column(DateTime(timezone=True), nullable=False)    # Candle open time (UTC)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)

    # Ensure no duplicate candles for the same symbol/timeframe/timestamp
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
    )


class TradeRecord(Base):
    """
    Database model for storing executed trades.
    Records every trade for audit trail and analysis.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(32), nullable=False, unique=True)     # Unique order identifier
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)                       # "buy" or "sell"
    order_type = Column(String(20), nullable=False)                 # "market", "limit", etc.
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)                           # Execution price
    commission = Column(Float, nullable=False, default=0.0)
    slippage = Column(Float, nullable=False, default=0.0)
    pnl = Column(Float, nullable=True)                              # Realized PnL (for closing trades)
    strategy = Column(String(50), nullable=True)                    # Strategy that generated the trade
    timestamp = Column(DateTime(timezone=True), nullable=False)
    notes = Column(String(500), nullable=True)


class DataStore:
    """
    Persistent data storage using SQLite.
    Manages OHLCV data caching and trade record storage.
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the data store with a SQLite database.

        Args:
            db_url: SQLAlchemy database URL. Defaults to local SQLite file.
        """
        if db_url is None:
            # Default database path in the project's data directory
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "trading.db")
            db_url = f"sqlite:///{db_path}"

        self.engine = create_engine(db_url, echo=False)
        # Create all tables if they don't exist
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"DataStore initialized with database: {db_url}")

    def store_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """
        Store OHLCV data to the database (upsert logic).

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1h")
            df: DataFrame with [open, high, low, close, volume] and DatetimeIndex

        Returns:
            Number of records stored/updated
        """
        count = 0
        session = self.SessionLocal()
        try:
            for idx, row in df.iterrows():
                # Check if this candle already exists
                existing = session.query(OHLCVRecord).filter(
                    OHLCVRecord.symbol == symbol,
                    OHLCVRecord.timeframe == timeframe,
                    OHLCVRecord.timestamp == idx,
                ).first()

                if existing:
                    # Update existing record
                    existing.open = row["open"]
                    existing.high = row["high"]
                    existing.low = row["low"]
                    existing.close = row["close"]
                    existing.volume = row["volume"]
                else:
                    # Insert new record
                    record = OHLCVRecord(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=idx,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                    session.add(record)
                count += 1

            session.commit()
            logger.debug(f"Stored {count} OHLCV records for {symbol} ({timeframe})")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store OHLCV data: {e}")
            raise
        finally:
            session.close()
        return count

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from the database.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Optional start filter
            end_time: Optional end filter

        Returns:
            DataFrame with OHLCV data and DatetimeIndex
        """
        session = self.SessionLocal()
        try:
            query = session.query(OHLCVRecord).filter(
                OHLCVRecord.symbol == symbol,
                OHLCVRecord.timeframe == timeframe,
            )
            if start_time:
                query = query.filter(OHLCVRecord.timestamp >= start_time)
            if end_time:
                query = query.filter(OHLCVRecord.timestamp <= end_time)

            query = query.order_by(OHLCVRecord.timestamp.asc())
            records = query.all()

            if not records:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            data = [{
                "datetime": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            } for r in records]

            df = pd.DataFrame(data)
            df.set_index("datetime", inplace=True)
            return df
        finally:
            session.close()

    def store_trade(self, trade: dict) -> None:
        """
        Store a single trade record to the database.

        Args:
            trade: Dictionary containing trade details
        """
        session = self.SessionLocal()
        try:
            record = TradeRecord(
                order_id=trade["order_id"],
                symbol=trade["symbol"],
                side=trade["side"],
                order_type=trade["order_type"],
                quantity=trade["quantity"],
                price=trade["price"],
                commission=trade.get("commission", 0.0),
                slippage=trade.get("slippage", 0.0),
                pnl=trade.get("pnl"),
                strategy=trade.get("strategy"),
                timestamp=trade.get("timestamp", datetime.now(timezone.utc)),
                notes=trade.get("notes"),
            )
            session.add(record)
            session.commit()
            logger.debug(f"Stored trade {trade['order_id']} for {trade['symbol']}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store trade: {e}")
            raise
        finally:
            session.close()

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Retrieve trade records from the database.

        Args:
            symbol: Optional filter by trading pair
            start_time: Optional start date filter
            end_time: Optional end date filter

        Returns:
            List of trade dictionaries
        """
        session = self.SessionLocal()
        try:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            if start_time:
                query = query.filter(TradeRecord.timestamp >= start_time)
            if end_time:
                query = query.filter(TradeRecord.timestamp <= end_time)

            query = query.order_by(TradeRecord.timestamp.asc())
            records = query.all()

            return [{
                "order_id": r.order_id,
                "symbol": r.symbol,
                "side": r.side,
                "order_type": r.order_type,
                "quantity": r.quantity,
                "price": r.price,
                "commission": r.commission,
                "slippage": r.slippage,
                "pnl": r.pnl,
                "strategy": r.strategy,
                "timestamp": r.timestamp,
                "notes": r.notes,
            } for r in records]
        finally:
            session.close()

    def get_trade_count(self, symbol: Optional[str] = None) -> int:
        """Get the total number of stored trades, optionally filtered by symbol."""
        session = self.SessionLocal()
        try:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            return query.count()
        finally:
            session.close()

    def clear_ohlcv(self, symbol: Optional[str] = None) -> None:
        """Delete all OHLCV records, optionally filtered by symbol."""
        session = self.SessionLocal()
        try:
            query = session.query(OHLCVRecord)
            if symbol:
                query = query.filter(OHLCVRecord.symbol == symbol)
            query.delete()
            session.commit()
            logger.info(f"Cleared OHLCV data{f' for {symbol}' if symbol else ''}")
        finally:
            session.close()
