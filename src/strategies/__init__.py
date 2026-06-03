"""Trading strategy module with multiple strategy implementations."""

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.ml_strategy import XGBoostStrategy, LSTMStrategy
from src.strategies.ensemble import EnsembleStrategy
