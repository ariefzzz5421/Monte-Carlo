"""Vectorized correlated GBM with month-end optional rebalancing."""

import numpy as np
from collections.abc import Callable

from src.config import Config
from src.portfolio import rebalance, should_rebalance


def simulate(
    config: Config,
    max_years: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Return portfolio values shaped (scenarios, months + 1).

    The configured annual return is an expected *simple* one-year return.
    log1p converts it into the GBM drift so E[S(1)/S(0)-1] equals that input.
    """
    months = max_years * 12
    rng = np.random.default_rng(config.seed)
    corr = np.asarray(config.correlation)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    sigma = np.asarray(config.annual_volatility)
    drift = np.log1p(config.annual_expected_returns) - 0.5 * sigma**2
    holdings = np.broadcast_to(
        config.initial_capital * np.asarray(config.weights),
        (config.simulations, 3),
    ).copy()
    paths = np.empty((config.simulations, months + 1), dtype=float)
    paths[:, 0] = config.initial_capital
    dt = 1 / 12
    for month in range(1, months + 1):
        shocks = rng.standard_normal((config.simulations, 3)) @ factor.T
        growth = np.exp(drift * dt + sigma * np.sqrt(dt) * shocks)
        holdings *= growth
        paths[:, month] = holdings.sum(axis=1)
        if should_rebalance(month, config.rebalancing):
            holdings = rebalance(holdings, np.asarray(config.weights))
        if progress_callback is not None:
            progress_callback(month, months)
    if not np.all(np.isfinite(paths)) or np.any(paths < 0):
        raise ValueError("simulation overflowed; choose less extreme assumptions")
    return paths
