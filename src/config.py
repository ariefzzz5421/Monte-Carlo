"""Load and validate an experiment before allocating large simulation arrays."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

ASSETS = ("BTC", "S&P 500", "Gold")
REBALANCING = ("none", "monthly", "annual")
REQUIRED = {
    "assumptions_note", "initial_capital", "weights", "annual_expected_returns",
    "annual_volatility", "correlation", "years", "simulations", "rebalancing", "seed",
}


@dataclass(frozen=True)
class Config:
    assumptions_note: str
    initial_capital: float
    weights: tuple[float, ...]
    annual_expected_returns: tuple[float, ...]
    annual_volatility: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    years: int
    simulations: int
    rebalancing: str
    seed: int

    def as_dict(self) -> dict:
        return {
            "assumptions_note": self.assumptions_note,
            "initial_capital": self.initial_capital,
            "weights": dict(zip(ASSETS, self.weights)),
            "annual_expected_returns": dict(zip(ASSETS, self.annual_expected_returns)),
            "annual_volatility": dict(zip(ASSETS, self.annual_volatility)),
            "correlation": [list(row) for row in self.correlation],
            "years": self.years,
            "simulations": self.simulations,
            "rebalancing": self.rebalancing,
            "seed": self.seed,
        }


def _real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _asset_values(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, dict) or set(value) != set(ASSETS):
        raise ValueError(f"{label} must contain exactly: {', '.join(ASSETS)}")
    return tuple(_real(value[asset], f"{label}.{asset}") for asset in ASSETS)


def validate_config(data: dict) -> Config:
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    missing, extra = REQUIRED - data.keys(), data.keys() - REQUIRED
    if missing or extra:
        raise ValueError(f"configuration keys: missing={sorted(missing)}, unknown={sorted(extra)}")
    note = data["assumptions_note"]
    if not isinstance(note, str) or not note.strip():
        raise ValueError("assumptions_note must be a nonempty string")
    capital = _real(data["initial_capital"], "initial_capital")
    if capital <= 0:
        raise ValueError("initial_capital must be positive")
    weights = _asset_values(data["weights"], "weights")
    if any(weight < 0 or weight > 1 for weight in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-8):
        raise ValueError("weights must each be between 0 and 1 and sum to 1")
    expected = _asset_values(data["annual_expected_returns"], "annual_expected_returns")
    if any(value <= -1 for value in expected):
        raise ValueError("annual_expected_returns must be greater than -1")
    volatility = _asset_values(data["annual_volatility"], "annual_volatility")
    if any(value < 0 for value in volatility):
        raise ValueError("annual_volatility cannot be negative")
    try:
        corr = np.asarray(data["correlation"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("correlation must be a numeric 3x3 matrix") from exc
    if corr.shape != (3, 3) or not np.all(np.isfinite(corr)):
        raise ValueError("correlation must be a finite 3x3 matrix")
    if not np.allclose(corr, corr.T, atol=1e-10, rtol=0):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(corr), 1, atol=1e-10, rtol=0):
        raise ValueError("correlation diagonal must be 1")
    if np.any(np.abs(corr) > 1 + 1e-10):
        raise ValueError("correlation coefficients must be between -1 and 1")
    if np.linalg.eigvalsh(corr).min() < -1e-10:
        raise ValueError("correlation must be positive semidefinite")
    years = _integer(data["years"], "years", 1)
    if years not in (1, 5, 10):
        raise ValueError("years must be 1, 5, or 10")
    simulations = _integer(data["simulations"], "simulations", 1)
    rebalancing = data["rebalancing"]
    if rebalancing not in REBALANCING:
        raise ValueError("rebalancing must be none, monthly, or annual")
    seed = _integer(data["seed"], "seed", 0)
    if seed > 2**64 - 1:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    return Config(note, capital, weights, expected, volatility,
                  tuple(tuple(float(x) for x in row) for row in corr),
                  years, simulations, rebalancing, seed)


def load_config(path: str | Path) -> Config:
    try:
        with Path(path).open(encoding="utf-8") as file:
            return validate_config(json.load(file))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
