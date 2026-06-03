"""
Strategy optimization and evaluation module.
Implements hyperparameter tuning, walk-forward validation,
and time-series cross-validation for robust strategy assessment.
"""

import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import BacktestConfig, StrategyConfig
from src.backtesting.engine import BacktestEngine
from src.strategies.base_strategy import BaseStrategy


class StrategyOptimizer:
    """
    Optimizes strategy parameters using grid search and walk-forward validation.

    Features:
        - Grid search over parameter space
        - Walk-forward validation to prevent overfitting
        - Time-series cross-validation (purged)
        - Strategy comparison across multiple configurations
        - Out-of-sample performance tracking
    """

    def __init__(self, backtest_config: Optional[BacktestConfig] = None):
        """
        Initialize the optimizer.

        Args:
            backtest_config: Backtesting configuration
        """
        self.bt_config = backtest_config or BacktestConfig()
        self.optimization_results: List[dict] = []

    def grid_search(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        data: Dict[str, pd.DataFrame],
        metric: str = "sharpe_ratio",
        initial_balance: float = 100000.0,
    ) -> List[dict]:
        """
        Perform grid search over strategy parameter space.

        Args:
            strategy_class: Strategy class to instantiate
            param_grid: Dict of parameter_name -> list of values to try
                        e.g., {"rsi_period": [10, 14, 20], "rsi_oversold": [25, 30]}
            data: Historical data dict for backtesting
            metric: Performance metric to optimize ("sharpe_ratio", "cagr_pct", etc.)
            initial_balance: Starting balance for each backtest

        Returns:
            List of results sorted by the target metric (best first)
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(
            f"Grid search: {len(combinations)} parameter combinations "
            f"for {strategy_class.__name__}"
        )

        results = []
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            # Create a strategy config with these parameters
            config = StrategyConfig()
            for key, value in params.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Instantiate strategy and run backtest
            strategy = strategy_class(config=config)
            engine = BacktestEngine(strategy, self.bt_config)

            try:
                result = engine.run(data, initial_balance)
                # Extract the target metric
                metric_value = self._extract_metric(result, metric)

                results.append({
                    "params": params,
                    "metric_name": metric,
                    "metric_value": metric_value,
                    "all_metrics": result["metrics"],
                })

                logger.debug(
                    f"  [{i + 1}/{len(combinations)}] {params} -> {metric}={metric_value:.4f}"
                )
            except Exception as e:
                logger.warning(f"  [{i + 1}/{len(combinations)}] {params} -> ERROR: {e}")
                results.append({
                    "params": params,
                    "metric_name": metric,
                    "metric_value": float("-inf"),
                    "error": str(e),
                })

        # Sort by target metric (descending = higher is better)
        results.sort(key=lambda r: r["metric_value"], reverse=True)
        self.optimization_results = results

        if results:
            best = results[0]
            logger.info(
                f"Best parameters: {best['params']} -> {metric}={best['metric_value']:.4f}"
            )

        return results

    def walk_forward_validation(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        window_size_days: Optional[int] = None,
        step_size_days: Optional[int] = None,
        initial_balance: float = 100000.0,
    ) -> dict:
        """
        Perform walk-forward validation.
        Splits data into overlapping train/test windows and evaluates
        out-of-sample performance to detect overfitting.

        Args:
            strategy: Strategy instance to validate
            data: Historical data dict
            window_size_days: Training window size in days
            step_size_days: Step size between windows in days
            initial_balance: Starting balance

        Returns:
            Dict with walk-forward results including in-sample and out-of-sample metrics
        """
        window_days = window_size_days or self.bt_config.walk_forward_window_days
        step_days = step_size_days or self.bt_config.walk_forward_step_days

        # Use the first symbol to determine date range
        primary_symbol = list(data.keys())[0]
        primary_df = data[primary_symbol]
        start_date = primary_df.index.min()
        end_date = primary_df.index.max()
        total_days = (end_date - start_date).days

        if total_days < window_days * 2:
            logger.warning("Insufficient data for walk-forward validation")
            return {"error": "insufficient_data"}

        fold_results = []
        fold_num = 0

        current_start = start_date

        while True:
            train_end = current_start + pd.Timedelta(days=window_days)
            test_end = train_end + pd.Timedelta(days=step_days)

            if test_end > end_date:
                break

            fold_num += 1

            # Split data into train and test sets
            train_data = {}
            test_data = {}
            for symbol, df in data.items():
                train_data[symbol] = df[(df.index >= current_start) & (df.index < train_end)]
                test_data[symbol] = df[(df.index >= train_end) & (df.index < test_end)]

            # Run backtest on train set (in-sample)
            strategy_copy = deepcopy(strategy)
            engine_train = BacktestEngine(strategy_copy, self.bt_config)
            train_result = engine_train.run(train_data, initial_balance)

            # Run backtest on test set (out-of-sample)
            strategy_copy2 = deepcopy(strategy)
            engine_test = BacktestEngine(strategy_copy2, self.bt_config)
            test_result = engine_test.run(test_data, initial_balance)

            fold_results.append({
                "fold": fold_num,
                "train_period": f"{current_start.date()} to {train_end.date()}",
                "test_period": f"{train_end.date()} to {test_end.date()}",
                "train_sharpe": self._extract_metric(train_result, "sharpe_ratio"),
                "test_sharpe": self._extract_metric(test_result, "sharpe_ratio"),
                "train_return": self._extract_metric(train_result, "total_return_pct"),
                "test_return": self._extract_metric(test_result, "total_return_pct"),
                "train_max_dd": self._extract_metric(train_result, "max_drawdown_pct"),
                "test_max_dd": self._extract_metric(test_result, "max_drawdown_pct"),
            })

            logger.info(
                f"  Fold {fold_num}: train_sharpe={fold_results[-1]['train_sharpe']:.4f}, "
                f"test_sharpe={fold_results[-1]['test_sharpe']:.4f}"
            )

            current_start += pd.Timedelta(days=step_days)

        if not fold_results:
            return {"error": "no_folds_generated"}

        # Aggregate walk-forward results
        avg_train_sharpe = np.mean([f["train_sharpe"] for f in fold_results])
        avg_test_sharpe = np.mean([f["test_sharpe"] for f in fold_results])
        avg_train_return = np.mean([f["train_return"] for f in fold_results])
        avg_test_return = np.mean([f["test_return"] for f in fold_results])

        # Overfitting ratio: how much performance degrades out-of-sample
        overfitting_ratio = (
            (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe)
            if avg_train_sharpe != 0 else 0
        )

        return {
            "folds": fold_results,
            "summary": {
                "total_folds": len(fold_results),
                "avg_train_sharpe": round(avg_train_sharpe, 4),
                "avg_test_sharpe": round(avg_test_sharpe, 4),
                "avg_train_return_pct": round(avg_train_return, 2),
                "avg_test_return_pct": round(avg_test_return, 2),
                "overfitting_ratio": round(overfitting_ratio, 4),
                "is_overfit": overfitting_ratio > 0.5,
            },
        }

    def time_series_cv(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        n_splits: int = 5,
        initial_balance: float = 100000.0,
    ) -> dict:
        """
        Perform time-series cross-validation with expanding window.

        Args:
            strategy: Strategy instance
            data: Historical data dict
            n_splits: Number of CV splits
            initial_balance: Starting balance

        Returns:
            Dict with CV results
        """
        primary_symbol = list(data.keys())[0]
        primary_df = data[primary_symbol]
        n_rows = len(primary_df)

        if n_rows < n_splits * 2:
            return {"error": "insufficient_data"}

        # Calculate split points
        test_size = n_rows // (n_splits + 1)
        fold_results = []

        for fold in range(n_splits):
            train_end_idx = test_size * (fold + 1)
            test_end_idx = train_end_idx + test_size

            if test_end_idx > n_rows:
                break

            train_data = {}
            test_data = {}
            for symbol, df in data.items():
                train_data[symbol] = df.iloc[:train_end_idx]
                test_data[symbol] = df.iloc[train_end_idx:test_end_idx]

            strategy_copy = deepcopy(strategy)
            engine = BacktestEngine(strategy_copy, self.bt_config)
            result = engine.run(test_data, initial_balance)

            sharpe = self._extract_metric(result, "sharpe_ratio")
            ret = self._extract_metric(result, "total_return_pct")
            max_dd = self._extract_metric(result, "max_drawdown_pct")

            fold_results.append({
                "fold": fold + 1,
                "sharpe": sharpe,
                "return_pct": ret,
                "max_drawdown_pct": max_dd,
            })

        if not fold_results:
            return {"error": "no_folds"}

        return {
            "folds": fold_results,
            "avg_sharpe": round(np.mean([f["sharpe"] for f in fold_results]), 4),
            "std_sharpe": round(np.std([f["sharpe"] for f in fold_results]), 4),
            "avg_return_pct": round(np.mean([f["return_pct"] for f in fold_results]), 2),
            "avg_max_dd_pct": round(np.mean([f["max_drawdown_pct"] for f in fold_results]), 2),
        }

    def compare_strategies(
        self,
        strategies: Dict[str, BaseStrategy],
        data: Dict[str, pd.DataFrame],
        initial_balance: float = 100000.0,
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data.

        Args:
            strategies: Dict of strategy_name -> strategy instance
            data: Historical data for backtesting
            initial_balance: Starting balance

        Returns:
            DataFrame with comparison metrics for each strategy
        """
        comparison = []

        for name, strategy in strategies.items():
            logger.info(f"Running backtest for strategy: {name}")
            engine = BacktestEngine(strategy, self.bt_config)
            result = engine.run(data, initial_balance)

            row = {
                "strategy": name,
                "total_return_pct": self._extract_metric(result, "total_return_pct"),
                "cagr_pct": self._extract_metric(result, "cagr_pct"),
                "sharpe_ratio": self._extract_metric(result, "sharpe_ratio"),
                "sortino_ratio": self._extract_metric(result, "sortino_ratio"),
                "max_drawdown_pct": self._extract_metric(result, "max_drawdown_pct"),
                "win_rate_pct": self._extract_metric(result, "win_rate_pct"),
                "profit_factor": self._extract_metric(result, "profit_factor"),
                "total_trades": self._extract_metric(result, "total_trades"),
            }
            comparison.append(row)

        df = pd.DataFrame(comparison)
        logger.info(f"\nStrategy Comparison:\n{df.to_string(index=False)}")
        return df

    @staticmethod
    def _extract_metric(result: dict, metric_name: str) -> float:
        """
        Extract a named metric from nested backtest results.

        Args:
            result: Backtest result dict
            metric_name: Metric name to extract

        Returns:
            Metric value as float
        """
        metrics = result.get("metrics", {})
        for category in metrics.values():
            if isinstance(category, dict) and metric_name in category:
                return float(category[metric_name])
        return 0.0
