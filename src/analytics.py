"""Scenario statistics with explicit terminal-loss conventions."""

import numpy as np


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Worst peak-to-trough loss in each path, as a positive fraction."""
    peaks = np.maximum.accumulate(paths, axis=1)
    return np.max(1 - paths / peaks, axis=1)


def summarize(paths: np.ndarray, initial: float, years: int) -> dict:
    finals = paths[:, -1]
    cagrs = np.power(finals / initial, 1 / years) - 1
    drawdowns = max_drawdowns(paths)
    losses = initial - finals  # USD loss; negative means terminal gain.
    var95 = float(np.percentile(losses, 95))
    tail = losses[losses >= var95]
    return {
        "horizon_years": years,
        "scenarios": int(len(finals)),
        "final_value_usd": {
            "mean": float(np.mean(finals)),
            "median": float(np.median(finals)),
            "p5": float(np.percentile(finals, 5)),
            "p95": float(np.percentile(finals, 95)),
        },
        "probability_below_initial": float(np.mean(finals < initial)),
        "probability_at_least_double": float(np.mean(finals >= 2 * initial)),
        "cagr": {
            "mean": float(np.mean(cagrs)),
            "median": float(np.median(cagrs)),
            "p5": float(np.percentile(cagrs, 5)),
            "p95": float(np.percentile(cagrs, 95)),
        },
        "max_drawdown": {
            "mean": float(np.mean(drawdowns)),
            "median": float(np.median(drawdowns)),
            "p5": float(np.percentile(drawdowns, 5)),
            "p95": float(np.percentile(drawdowns, 95)),
        },
        "terminal_loss_risk_usd": {
            "convention": "loss = initial capital - final value; negative is a gain",
            "var_95": var95,
            "expected_shortfall_95": float(np.mean(tail)),
            "tail_scenarios": int(len(tail)),
        },
    }


def percentile_paths(paths: np.ndarray) -> np.ndarray:
    """Rows correspond to month 0..N, columns are P5, median, P95."""
    return np.percentile(paths, [5, 50, 95], axis=0).T
