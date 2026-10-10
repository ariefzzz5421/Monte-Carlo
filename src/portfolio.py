"""Portfolio holdings and rebalancing operations."""

import numpy as np


def rebalance(holdings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Invest each scenario's full current portfolio value at target weights."""
    return holdings.sum(axis=1, keepdims=True) * weights[np.newaxis, :]


def should_rebalance(month: int, frequency: str) -> bool:
    return frequency == "monthly" or (frequency == "annual" and month % 12 == 0)
