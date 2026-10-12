"""Arbitrary-asset spot portfolios with GBM or daily block bootstrap.

Historical estimates are descriptive. GBM always uses a separate set of
explicit, hypothetical future assumptions; bootstrap uses only imported data.
"""

from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
import math

import numpy as np

from src.analytics import summarize
from src.portfolio import should_rebalance


DEFAULT_SPOT = {
    "initial_capital": 10_000.0,
    "assets": ["BTC", "HYPE", "USDT", "XAUT"],
    "weights": {"BTC": 0.50, "HYPE": 0.20, "USDT": 0.15, "XAUT": 0.15},
    "years": 5, "runs": 10_000, "seed": 42, "rebalancing": "monthly",
    "method": "gbm", "block_days": 7,
    "future_assumptions_note": "Hypothetical educational inputs, not historical estimates or forecasts.",
    "future_expected_returns": {"BTC": 0.08, "HYPE": 0.12, "USDT": 0.0, "XAUT": 0.04},
    "future_volatility": {"BTC": 0.60, "HYPE": 0.80, "USDT": 0.005, "XAUT": 0.18},
    "future_correlation": [[1.0, 0.45, 0.0, 0.10], [0.45, 1.0, 0.0, 0.05],
                           [0.0, 0.0, 1.0, 0.0], [0.10, 0.05, 0.0, 1.0]],
    "stablecoin_depeg": {"enabled": False, "asset": "USDT", "month": 12, "multiplier": 0.8},
    "crypto_crash": {"enabled": False, "assets": ["BTC", "HYPE"], "month": 12, "multiplier": 0.5},
}


@dataclass(frozen=True)
class SpotConfig:
    data: dict

    def as_dict(self) -> dict:
        return self.data


@dataclass(frozen=True)
class PriceHistory:
    dates: tuple[date, ...]
    assets: tuple[str, ...]
    prices: np.ndarray

    @property
    def log_returns(self) -> np.ndarray:
        return np.diff(np.log(self.prices), axis=0)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def validate_spot_config(data: dict) -> SpotConfig:
    if not isinstance(data, dict) or set(data) != set(DEFAULT_SPOT):
        raise ValueError("spot configuration must contain exactly the documented JSON keys")
    assets = data["assets"]
    if not isinstance(assets, list) or not 1 <= len(assets) <= 30 or any(
        not isinstance(a, str) or not a.strip() or a != a.strip() for a in assets
    ) or len(set(assets)) != len(assets):
        raise ValueError("assets must be 1–30 unique, nonblank names")
    capital = _number(data["initial_capital"], "initial_capital")
    if capital <= 0:
        raise ValueError("initial_capital must be positive")
    weights = data["weights"]
    if not isinstance(weights, dict) or set(weights) != set(assets):
        raise ValueError("weights must include each asset exactly once")
    w = np.array([_number(weights[a], f"weight {a}") for a in assets])
    if np.any((w < 0) | (w > 1)) or not math.isclose(float(w.sum()), 1, abs_tol=1e-8):
        raise ValueError("portfolio weights must be within 0–1 and sum to 1")
    _integer(data["years"], "years", 1, 10)
    _integer(data["runs"], "runs", 1, 100_000)
    _integer(data["seed"], "seed", 0, 2**32 - 1)
    _integer(data["block_days"], "block_days", 1, 365)
    if data["rebalancing"] not in ("none", "monthly", "annual"):
        raise ValueError("rebalancing must be none, monthly or annual")
    if data["method"] not in ("gbm", "block_bootstrap"):
        raise ValueError("method must be gbm or block_bootstrap")
    note = data["future_assumptions_note"]
    if not isinstance(note, str) or not note.strip():
        raise ValueError("future_assumptions_note must describe the hypothetical inputs")
    expected, volatility = data["future_expected_returns"], data["future_volatility"]
    if not isinstance(expected, dict) or not isinstance(volatility, dict) or set(expected) != set(assets) or set(volatility) != set(assets):
        raise ValueError("future return and volatility assumptions must include each asset")
    mu = np.array([_number(expected[a], f"future return {a}") for a in assets])
    sigma = np.array([_number(volatility[a], f"future volatility {a}") for a in assets])
    if np.any(mu <= -1) or np.any(sigma < 0):
        raise ValueError("future returns must exceed -100% and volatility cannot be negative")
    try:
        corr = np.asarray(data["future_correlation"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("future_correlation must be numeric") from exc
    n = len(assets)
    if corr.shape != (n, n) or not np.all(np.isfinite(corr)):
        raise ValueError("future_correlation must be a finite NxN matrix in asset order")
    if not np.allclose(corr, corr.T, atol=1e-10, rtol=0) or not np.allclose(np.diag(corr), 1, atol=1e-10, rtol=0):
        raise ValueError("future_correlation must be symmetric with diagonal 1")
    if np.any(np.abs(corr) > 1 + 1e-10) or np.linalg.eigvalsh(corr).min() < -1e-10:
        raise ValueError("future_correlation must be a valid positive semidefinite correlation matrix")
    depeg = data["stablecoin_depeg"]
    crash = data["crypto_crash"]
    if not isinstance(depeg, dict) or set(depeg) != {"enabled", "asset", "month", "multiplier"}:
        raise ValueError("stablecoin_depeg needs enabled, asset, month, multiplier")
    if not isinstance(crash, dict) or set(crash) != {"enabled", "assets", "month", "multiplier"}:
        raise ValueError("crypto_crash needs enabled, assets, month, multiplier")
    for label, stress in (("stablecoin_depeg", depeg), ("crypto_crash", crash)):
        if not isinstance(stress["enabled"], bool):
            raise ValueError(f"{label}.enabled must be Boolean")
        _integer(stress["month"], f"{label}.month", 1, data["years"] * 12)
        multiplier = _number(stress["multiplier"], f"{label}.multiplier")
        if not 0 <= multiplier <= 1:
            raise ValueError(f"{label}.multiplier must be between 0 and 1")
    if not isinstance(depeg["asset"], str) or (depeg["enabled"] and depeg["asset"] not in assets):
        raise ValueError("depeg asset must be in the portfolio when enabled")
    if (not isinstance(crash["assets"], list)
        or any(not isinstance(asset, str) for asset in crash["assets"])
        or (crash["enabled"] and (
            not crash["assets"] or len(set(crash["assets"])) != len(crash["assets"])
            or not set(crash["assets"]).issubset(assets)
        ))):
        raise ValueError("crash assets must be a nonempty subset of portfolio assets when enabled")
    return SpotConfig(data)


def parse_price_csv(content: str | bytes, assets: tuple[str, ...]) -> PriceHistory:
    """Read wide daily USD prices: date, one column per asset, no gaps."""
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("price CSV must be UTF-8") from exc
    reader = csv.DictReader(StringIO(content))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != {"date", *assets}:
        raise ValueError("price CSV must have date and exactly one column for each configured asset")
    dates, prices = [], []
    for line, row in enumerate(reader, 2):
        if None in row or any(value is None or value == "" for value in row.values()):
            raise ValueError(f"price CSV line {line}: every asset needs an aligned price")
        try:
            day = date.fromisoformat(row["date"])
        except ValueError as exc:
            raise ValueError(f"price CSV line {line}: date must be YYYY-MM-DD") from exc
        if dates and day != dates[-1] + timedelta(days=1):
            raise ValueError(f"price CSV line {line}: dates must be unique, ascending, contiguous calendar days")
        values = [_number(row[asset], f"line {line} {asset}") for asset in assets]
        if any(value <= 0 for value in values):
            raise ValueError(f"price CSV line {line}: USD prices must be positive")
        dates.append(day)
        prices.append(values)
    if len(dates) < 31:
        raise ValueError("price CSV needs at least 31 aligned daily observations")
    return PriceHistory(tuple(dates), assets, np.asarray(prices, dtype=float))


def estimate_history(history: PriceHistory) -> dict:
    """Descriptive estimates from observed adjacent daily log returns, 365 days/year."""
    logs = history.log_returns
    cov = np.atleast_2d(np.cov(logs, rowvar=False, ddof=1)) * 365
    vol = np.sqrt(np.maximum(np.diag(cov), 0))
    denominator = np.outer(vol, vol)
    # Near-constant columns make Pearson correlation undefined and magnify
    # floating-point roundoff; display zero off-diagonal by convention.
    usable = vol > 1e-12
    corr = np.divide(cov, denominator, out=np.zeros_like(cov),
                     where=(denominator > 0) & np.outer(usable, usable))
    np.fill_diagonal(corr, 1)
    return {
        "label": "historical descriptive estimates from imported USD prices; not future assumptions",
        "observations": len(history.dates), "first_date": history.dates[0].isoformat(),
        "last_date": history.dates[-1].isoformat(), "assets": list(history.assets),
        "mean_daily_log_return": logs.mean(axis=0).tolist(),
        "annualized_log_return_mean": (logs.mean(axis=0) * 365).tolist(),
        "annualized_volatility": vol.tolist(), "annualized_covariance": cov.tolist(),
        "historical_correlation": corr.tolist(),
        "undefined_correlation_assets": [asset for asset, valid in zip(history.assets, usable) if not valid],
    }


def simulate_spot(
    config: SpotConfig, history: PriceHistory | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    data = config.data
    assets = data["assets"]
    if data["method"] == "block_bootstrap" and history is None:
        raise ValueError("block bootstrap requires imported daily price CSV")
    if history is not None and history.assets != tuple(assets):
        raise ValueError("price CSV assets do not match the portfolio")
    if history is not None and data["block_days"] > len(history.log_returns):
        raise ValueError("block_days cannot exceed historical daily return observations")
    runs, months = data["runs"], data["years"] * 12
    weights = np.asarray([data["weights"][a] for a in assets])
    holdings = np.broadcast_to(data["initial_capital"] * weights, (runs, len(assets))).copy()
    paths = np.empty((runs, months + 1), dtype=float)
    paths[:, 0] = data["initial_capital"]
    rng = np.random.default_rng(data["seed"])
    if data["method"] == "gbm":
        corr = np.asarray(data["future_correlation"])
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        sigma = np.asarray([data["future_volatility"][a] for a in assets])
        drift = np.log1p([data["future_expected_returns"][a] for a in assets]) - 0.5 * sigma**2
    else:
        logs = history.log_returns
        block_days = data["block_days"]
    checkpoints = np.rint(np.linspace(0, data["years"] * 365, months + 1)).astype(int)
    for month in range(1, months + 1):
        if data["method"] == "gbm":
            shocks = rng.standard_normal((runs, len(assets))) @ factor.T
            growth = np.exp(drift / 12 + sigma / np.sqrt(12) * shocks)
        else:
            days = checkpoints[month] - checkpoints[month - 1]
            block_count = math.ceil(days / block_days)
            starts = rng.integers(0, len(logs) - block_days + 1, size=(runs, block_count))
            offsets = np.arange(block_days)
            indices = (starts[:, :, None] + offsets).reshape(runs, -1)[:, :days]
            growth = np.exp(logs[indices].sum(axis=1))
        holdings *= growth
        depeg = data["stablecoin_depeg"]
        if depeg["enabled"] and month == depeg["month"]:
            holdings[:, assets.index(depeg["asset"])] *= depeg["multiplier"]
        crash = data["crypto_crash"]
        if crash["enabled"] and month == crash["month"]:
            holdings[:, [assets.index(a) for a in crash["assets"]]] *= crash["multiplier"]
        paths[:, month] = holdings.sum(axis=1)
        if should_rebalance(month, data["rebalancing"]):
            holdings = paths[:, month, None] * weights
        if progress_callback is not None:
            progress_callback(month, months)
    if not np.all(np.isfinite(paths)) or np.any(paths < 0):
        raise ValueError("spot simulation overflowed; reduce extreme assumptions")
    return paths


def summarize_spot(paths: np.ndarray, config: SpotConfig, history: PriceHistory | None) -> dict:
    data = config.data
    return {
        "mode": "spot_portfolio", "method": data["method"],
        "assumptions": data, "historical_estimates": estimate_history(history) if history else None,
        "source_label": "imported historical USD prices" if history else "no historical data imported",
        "selected_horizon": summarize(paths, data["initial_capital"], data["years"]),
    }
