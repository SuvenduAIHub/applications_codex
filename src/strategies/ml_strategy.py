"""
Machine Learning-based trading strategy.
Implements XGBoost for feature-based prediction and LSTM (PyTorch) for
sequential price prediction. Provides ensemble ML signals.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, TradeSignal

# Lazy imports for ML libraries (may not be installed in all environments)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed; XGBoost strategy disabled")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed; LSTM strategy disabled")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# LSTM Model Definition (PyTorch)
# ============================================================

if HAS_TORCH:
    class LSTMPredictor(nn.Module):
        """
        LSTM neural network for price direction prediction.
        Takes a sequence of feature vectors and outputs a buy/sell probability.
        """

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            """
            Initialize the LSTM model.

            Args:
                input_size: Number of input features per timestep
                hidden_size: LSTM hidden state dimension
                num_layers: Number of stacked LSTM layers
                dropout: Dropout rate for regularization
            """
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM encoder for sequential data
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            # Fully connected output layer: predicts probability of price going up
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),  # Output probability between 0 and 1
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through LSTM and FC layers."""
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Use the last timestep's output
            last_hidden = lstm_out[:, -1, :]
            output = self.fc(last_hidden)
            return output


# ============================================================
# XGBoost Strategy
# ============================================================

class XGBoostStrategy(BaseStrategy):
    """
    XGBoost-based trading strategy.
    Uses gradient-boosted trees to predict price direction
    from technical indicator features.
    """

    def __init__(self, config=None, sequence_length: int = 10):
        """
        Initialize the XGBoost strategy.

        Args:
            config: Strategy configuration
            sequence_length: Number of historical bars used as features
        """
        super().__init__(config)
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.feature_columns: List[str] = []

    def get_name(self) -> str:
        return "xgboost_ml"

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Select the feature columns from the DataFrame for model input.
        Excludes raw OHLCV and non-numeric columns.
        """
        exclude = {"open", "high", "low", "close", "volume", "regime", "macro_regime",
                    "regime_str", "macro_regime_str"}
        return [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare feature matrix (X) and optional target (y) from DataFrame.
        Target: 1 if price goes up in next candle, 0 otherwise.

        Returns:
            Tuple of (X features array, y target array or None)
        """
        if not self.feature_columns:
            self.feature_columns = self._get_feature_columns(df)

        # Select feature columns that exist in the DataFrame
        available = [c for c in self.feature_columns if c in df.columns]
        feature_df = df[available].copy()

        # Fill NaN values with forward/backward fill, then zero
        feature_df = feature_df.ffill().bfill().fillna(0)

        X = feature_df.values

        # Create target: 1 if next close > current close
        if "close" in df.columns and len(df) > 1:
            y = (df["close"].shift(-1) > df["close"]).astype(int).values
            # Remove the last row (no target available)
            X = X[:-1]
            y = y[:-1]
        else:
            y = None

        return X, y

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the XGBoost model on historical data.

        Args:
            df: DataFrame with indicators for training

        Returns:
            Dict with training metrics (accuracy, etc.)
        """
        if not HAS_XGBOOST:
            logger.error("XGBoost not available for training")
            return {"error": "XGBoost not installed"}

        X, y = self._prepare_features(df)
        if y is None or len(X) < 100:
            logger.warning("Insufficient data for XGBoost training")
            return {"error": "insufficient_data"}

        # Scale features for better model performance
        if self.scaler:
            X = self.scaler.fit_transform(X)

        # Time-series aware cross-validation split
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Train XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,        # L1 regularization to prevent overfitting
            reg_lambda=1.0,       # L2 regularization
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Calculate validation metrics
        val_pred = self.model.predict(X_val)
        accuracy = float(np.mean(val_pred == y_val))
        self.is_trained = True

        logger.info(f"XGBoost model trained: validation accuracy = {accuracy:.4f}")
        return {
            "accuracy": accuracy,
            "train_size": len(X_train),
            "val_size": len(X_val),
        }

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate a trading signal using the trained XGBoost model.

        Args:
            df: DataFrame with indicators
            symbol: Trading pair symbol

        Returns:
            TradeSignal based on XGBoost prediction
        """
        price = df.iloc[-1]["close"] if len(df) > 0 else 0.0

        if not self.is_trained or self.model is None:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD, confidence=0.0,
                strategy_name=self.get_name(), price=price,
                metadata={"reason": "model_not_trained"},
            )

        # Prepare the latest features for prediction (use only training-time columns)
        available = [c for c in self.feature_columns if c in df.columns]
        feature_row = df[available].iloc[-1:].fillna(0).values

        # If feature count mismatch (e.g., data has regime columns added after training),
        # re-fit scaler or skip prediction to avoid errors
        if self.scaler:
            expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else feature_row.shape[1]
            if feature_row.shape[1] != expected_features:
                # Feature mismatch — generate neutral signal instead of crashing
                return TradeSignal(
                    symbol=symbol, signal=Signal.HOLD, confidence=0.0,
                    strategy_name=self.get_name(), price=price,
                    metadata={"reason": "feature_mismatch", "expected": expected_features, "got": feature_row.shape[1]},
                )
            feature_row = self.scaler.transform(feature_row)

        # Get prediction probability
        prob = self.model.predict_proba(feature_row)[0]
        prob_up = float(prob[1]) if len(prob) > 1 else 0.5

        atr = df.iloc[-1].get("atr", price * 0.02)

        # Convert probability to signal
        if prob_up > 0.65:
            signal = Signal.STRONG_BUY if prob_up > 0.8 else Signal.BUY
            return TradeSignal(
                symbol=symbol, signal=signal, confidence=prob_up,
                strategy_name=self.get_name(), price=price,
                stop_loss=price - (2 * atr), take_profit=price + (3 * atr),
                metadata={"prob_up": prob_up, "prob_down": 1 - prob_up},
            )
        elif prob_up < 0.35:
            signal = Signal.STRONG_SELL if prob_up < 0.2 else Signal.SELL
            return TradeSignal(
                symbol=symbol, signal=signal, confidence=1 - prob_up,
                strategy_name=self.get_name(), price=price,
                stop_loss=price + (2 * atr), take_profit=price - (3 * atr),
                metadata={"prob_up": prob_up, "prob_down": 1 - prob_up},
            )

        return TradeSignal(
            symbol=symbol, signal=Signal.HOLD, confidence=0.0,
            strategy_name=self.get_name(), price=price,
            metadata={"prob_up": prob_up},
        )


# ============================================================
# LSTM Strategy
# ============================================================

class LSTMStrategy(BaseStrategy):
    """
    LSTM-based trading strategy using PyTorch.
    Processes sequences of indicator features to predict price direction.
    """

    def __init__(self, config=None, sequence_length: int = 30, hidden_size: int = 64):
        """
        Initialize the LSTM strategy.

        Args:
            config: Strategy configuration
            sequence_length: Number of timesteps in each input sequence
            hidden_size: LSTM hidden dimension
        """
        super().__init__(config)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.feature_columns: List[str] = []

    def get_name(self) -> str:
        return "lstm_ml"

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select numeric feature columns for LSTM input."""
        exclude = {"open", "high", "low", "close", "volume", "regime", "macro_regime",
                    "regime_str", "macro_regime_str"}
        return [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]

    def _prepare_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences of features for LSTM input.

        Returns:
            Tuple of (X sequences array of shape [n, seq_len, features], y targets)
        """
        if not self.feature_columns:
            self.feature_columns = self._get_feature_columns(df)

        available = [c for c in self.feature_columns if c in df.columns]
        feature_df = df[available].ffill().bfill().fillna(0)
        values = feature_df.values

        if self.scaler:
            values = self.scaler.fit_transform(values)

        # Create sliding window sequences
        X_sequences = []
        y_targets = []
        for i in range(self.sequence_length, len(values)):
            X_sequences.append(values[i - self.sequence_length:i])
            # Target: 1 if next close > current close
            if i < len(df) - 1:
                y_targets.append(1 if df["close"].iloc[i + 1] > df["close"].iloc[i] else 0)

        X = np.array(X_sequences)
        # Align X and y lengths
        min_len = min(len(X), len(y_targets))
        X = X[:min_len]
        y = np.array(y_targets[:min_len]) if y_targets else None

        return X, y

    def train(self, df: pd.DataFrame, epochs: int = 50, lr: float = 0.001) -> Dict[str, float]:
        """
        Train the LSTM model on historical data.

        Args:
            df: DataFrame with indicators
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Dict with training metrics
        """
        if not HAS_TORCH:
            logger.error("PyTorch not available for LSTM training")
            return {"error": "PyTorch not installed"}

        X, y = self._prepare_sequences(df)
        if y is None or len(X) < 100:
            logger.warning("Insufficient data for LSTM training")
            return {"error": "insufficient_data"}

        # Train/validation split (time-series aware: no shuffling)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        # Initialize model
        input_size = X_train.shape[2]
        self.model = LSTMPredictor(input_size=input_size, hidden_size=self.hidden_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Training loop
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            output = self.model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_loss = criterion(val_output, y_val_t).item()
                val_pred = (val_output > 0.5).float()
                val_acc = float((val_pred == y_val_t).float().mean())

            # Early stopping to prevent overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"LSTM early stopping at epoch {epoch + 1}")
                    break

        self.is_trained = True
        logger.info(f"LSTM model trained: val_accuracy = {val_acc:.4f}, val_loss = {best_val_loss:.4f}")
        return {
            "val_accuracy": val_acc,
            "val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
        }

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        Generate a trading signal using the trained LSTM model.

        Args:
            df: DataFrame with indicators
            symbol: Trading pair symbol

        Returns:
            TradeSignal based on LSTM prediction
        """
        price = df.iloc[-1]["close"] if len(df) > 0 else 0.0

        if not self.is_trained or self.model is None or not HAS_TORCH:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD, confidence=0.0,
                strategy_name=self.get_name(), price=price,
                metadata={"reason": "model_not_trained"},
            )

        # Prepare the latest sequence
        available = [c for c in self.feature_columns if c in df.columns]
        feature_df = df[available].iloc[-self.sequence_length:].fillna(0)

        if len(feature_df) < self.sequence_length:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD, confidence=0.0,
                strategy_name=self.get_name(), price=price,
                metadata={"reason": "insufficient_sequence_length"},
            )

        values = feature_df.values
        if self.scaler:
            values = self.scaler.transform(values)

        # Run inference
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(values).unsqueeze(0)  # Add batch dimension
            prob_up = float(self.model(x).item())

        atr = df.iloc[-1].get("atr", price * 0.02)

        if prob_up > 0.6:
            signal = Signal.STRONG_BUY if prob_up > 0.75 else Signal.BUY
            return TradeSignal(
                symbol=symbol, signal=signal, confidence=prob_up,
                strategy_name=self.get_name(), price=price,
                stop_loss=price - (2 * atr), take_profit=price + (3 * atr),
                metadata={"prob_up": prob_up},
            )
        elif prob_up < 0.4:
            signal = Signal.STRONG_SELL if prob_up < 0.25 else Signal.SELL
            return TradeSignal(
                symbol=symbol, signal=signal, confidence=1 - prob_up,
                strategy_name=self.get_name(), price=price,
                stop_loss=price + (2 * atr), take_profit=price - (3 * atr),
                metadata={"prob_up": prob_up},
            )

        return TradeSignal(
            symbol=symbol, signal=Signal.HOLD, confidence=0.0,
            strategy_name=self.get_name(), price=price,
            metadata={"prob_up": prob_up},
        )
